#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from llava.model.language_model.llava_mpt import LlavaMptForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM



def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs
):
    """
    Loads a LLaVA or standard language model checkpoint from local path or HF repo,
    merges LoRA weights if needed, and patches the config for attention_dropout.
    """
    # Step 1: prepare the device map in kwargs
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    # Step 2: handle quantization arguments
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Step 3: optionally use flash attention
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # Step 4: If the model name indicates "llava"
    if 'llava' in model_name.lower():
        # 4a: If LoRA
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. "
                "If you are loading a LoRA model, please provide the `model_base` argument. "
                "Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )

        if 'lora' in model_name.lower() and model_base is not None:
            # LoRA path
            print('Loading LLaVA LoRA from base model...')
            # Example: lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            if not hasattr(model.config, "attention_dropout"):
              model.config.attention_dropout = 0.1  # or any default you like
              print(f"Patched model.config.attention_dropout = {model.config.attention_dropout}")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                **kwargs
            )

            # Possibly fix output embeddings if mismatch
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )

            print('Loading additional LLaVA weights...')
            non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
            else:
                # Attempt from HF Hub if not local
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder
                    )
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

            # Tweak keys if they have prefixes
            non_lora_trainables = {
                (k[11:] if k.startswith('base_model.') else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith('model.') else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

        elif model_base is not None:
            # base LLaVA with mm_projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                # MPT-based
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(
                        os.path.join(model_base, 'configuration_mpt.py'),
                        os.path.join(model_path, 'configuration_mpt.py')
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
               if not hasattr(model.config, "attention_dropout"):
                 model.config.attention_dropout = 0.1  # or any default you like
                 print(f"Patched model.config.attention_dropout = {model.config.attention_dropout}")
                model = LlavaMptForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
            else:
                # LLaMA-based
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )

            # load mm_projector
            projector_path = os.path.join(model_path, 'mm_projector.bin')
            mm_projector_weights = torch.load(projector_path, map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)

        else:
            # load from model_path directly
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
              if not hasattr(model.config, "attention_dropout"):
                 model.config.attention_dropout = 0.1  # or any default you like
                 print(f"Patched model.config.attention_dropout = {model.config.attention_dropout}")
               model = LlavaMptForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if not hasattr(model.config, "attention_dropout"):
              model.config.attention_dropout = 0.1  # or any default you like
              print(f"Patched model.config.attention_dropout = {model.config.attention_dropout}")
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )

    else:
        # Standard language model (not llava)
        if model_base is not None:
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            # No base model provided
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )

    # Patch attention_dropout if missing
    if not hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = 0.1  # or any default you like
        print(f"Patched model.config.attention_dropout = {model.config.attention_dropout}")

    # Possibly load image processor if it's LLaVA
    image_processor = None
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        # get vision tower
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    # set context length
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
