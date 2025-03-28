import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mpt import LlavaMptForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava_config import LlavaConfig
from transformers import AutoTokenizer, CLIPImageProcessor
from llava.model.llava_llama import LlavaLlamaForCausalLM


def load_pretrained_model(model_path, model_base, model_name, mm_projector_type="qformer",
                          load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", use_flash_attn=False, **kwargs):

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

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

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_base or model_path, use_fast=False)

        # Load custom LlavaConfig explicitly
        cfg_pretrained = LlavaConfig.from_pretrained(model_path)
        cfg_pretrained.mm_projector_type = mm_projector_type

        if model_base is not None:
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            # Load mm_projector weights explicitly
            projector_path = os.path.join(model_path, 'mm_projector.bin')
            if os.path.exists(projector_path):
                mm_projector_weights = torch.load(projector_path, map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)

            # LoRA handling
            if 'lora' in model_name.lower():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_path)
                model = model.merge_and_unload()
        else:
            # Direct LLaVA load
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    else:
        # Non-LLaVA models
        tokenizer = AutoTokenizer.from_pretrained(model_base or model_path, use_fast=False)
        if model_base is not None:
            from peft import PeftModel
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            model.to(torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # Image processor and embedding resizing for LLaVA
    image_processor = None
    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    context_len = getattr(model.config, "max_sequence_length", 2048)

    return tokenizer, model, image_processor, context_len


