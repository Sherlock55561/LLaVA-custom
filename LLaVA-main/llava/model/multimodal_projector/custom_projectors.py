from transformers import Blip2Model
import torch.nn as nn
import torch

class QFormerProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load a pretrained BLIP-2 model (adjust model name as desired)
        blip2 = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        # Use its vision encoder and Q-Former parts.
        self.vision_encoder = blip2.vision_model  # CLIP ViT encoder
        self.qformer = blip2.qformer              # Q-Former transformer

        # Freeze vision encoder parameters if desired:
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # The Q-Former outputs query embeddings; we then project them to LLM embed size.
        q_hidden_size = self.qformer.config.hidden_size  # e.g. 768
        self.proj = nn.Linear(q_hidden_size, config.hidden_size)

    def forward(self, images):
        # images: preprocessed image tensor(s)
        # 1. Get patch embeddings from the vision encoder
        vis_feats = self.vision_encoder(images).last_hidden_state  # shape: (B, N_patches, vis_dim)
        # 2. Pass the image features through the Q-Former.
        # The Q-Former uses a set of learnable queries to attend over image patches.
        q_outputs = self.qformer(encoder_hidden_states=vis_feats)
        # q_outputs.last_hidden_state has shape: (B, num_queries, q_hidden_size)
        query_embeds = q_outputs.last_hidden_state  
        # 3. Project the query embeddings to the LLM's hidden dimension.
        projected = self.proj(query_embeds)  # shape: (B, num_queries, hidden_size)
        return projected
    
class HybridProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Global projection using a linear layer applied to the CLS token
        self.global_proj = nn.Linear(config.mm_hidden_size, config.hidden_size)
        
        # Q-Former branch: using a pre-trained BLIP-2 Q-Former (as above)
        blip2 = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.vision_encoder = blip2.vision_model
        self.qformer = blip2.qformer
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Project Q-Former output to LLM dimension
        q_hidden_size = self.qformer.config.hidden_size
        self.q_proj = nn.Linear(q_hidden_size, config.hidden_size)
    
    def forward(self, images):
        # Extract image features from vision encoder.
        vis_feats = self.vision_encoder(images).last_hidden_state  # (B, N_patches, mm_hidden_size)
        # Assume that the first token (index 0) is the global (CLS) token.
        global_feat = vis_feats[:, 0, :]  # (B, mm_hidden_size)
        global_emb = self.global_proj(global_feat)  # (B, hidden_size)
        
        # Q-Former branch: process all patch tokens.
        q_outputs = self.qformer(encoder_hidden_states=vis_feats)
        # Get a fixed set of query embeddings (e.g. 8 tokens).
        q_embeds = q_outputs.last_hidden_state  # (B, num_queries, q_hidden_size)
        q_embeds = self.q_proj(q_embeds)  # (B, num_queries, hidden_size)
        
        # Concatenate global embedding (as a token) with Q-Former outputs.
        global_emb = global_emb.unsqueeze(1)  # (B, 1, hidden_size)
        combined = torch.cat([global_emb, q_embeds], dim=1)  # (B, 1+num_queries, hidden_size)
        return combined