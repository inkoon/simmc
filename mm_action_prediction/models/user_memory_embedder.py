"""Embedding user memory for action prediction for fashion.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from tools import torch_support as support


class UserMemoryEmbedder(nn.Module):
    def __init__(self, params):
        super(UserMemoryEmbedder, self).__init__()
        self.params = params
        self.host = torch.cuda if params["use_gpu"] else torch
        self.categories = ["focus", "database", "memory"]
        self.category_embeds = {}
        for position in self.categories:
            pos_parameter = torch.randn(params["word_embed_size"])
            if params["use_gpu"]:
                pos_parameter = pos_parameter.cuda()
            pos_parameter = nn.Parameter(pos_parameter)
            self.category_embeds[position] = pos_parameter
            # Register the parameter for training/saving.
            self.register_parameter(position, pos_parameter)
        self.category_state = None
        # Project multimodal embedding to same size as encoder.
        input_size = params["asset_feature_size"] + params["word_embed_size"]
        if params["text_encoder"] == "lstm":
            output_size = params["hidden_size"]
        else:
            output_size = params["word_embed_size"]
        self.multimodal_embed_net = nn.Linear(input_size, output_size)
        self.multimodal_attend = nn.MultiheadAttention(output_size, 1)

        self.MAG = Multimodal_Adaptation_Gate(params)

    def forward(self, multimodal_state, encoder_state, encoder_size):
        """Multimodal Embedding.

        Args:
            multimodal_state: Dict with memory, database, and focus images
            encoder_state: State of the encoder
            encoder_size: (batch_size, num_rounds)

        Returns:
            multimodal_encode: Encoder state with multimodal information
        """
        # Setup category states if None.
        if self.category_state is None:
            self._setup_category_states()
        # Attend to multimodal memory using encoder states.
        batch_size, num_rounds = encoder_size
        memory_images = multimodal_state["memory_images"]
        memory_images = memory_images.unsqueeze(1).expand(-1, num_rounds, -1, -1)
        focus_images = multimodal_state["focus_images"][:, :num_rounds, :]
        focus_images = focus_images.unsqueeze(2)
        all_images = torch.cat([focus_images, memory_images], dim=2)
        all_images_flat = support.flatten(all_images, batch_size, num_rounds)
        category_state = self.category_state.expand(batch_size * num_rounds, -1, -1)
        cat_images = torch.cat([all_images_flat, category_state], dim=-1)
        multimodal_memory = self.multimodal_embed_net(cat_images)
        # Key (L, N, E), value (L, N, E), query (S, N, E)
        multimodal_memory = multimodal_memory.transpose(0, 1)
        query = encoder_state.unsqueeze(0)

        #MAG
        multimodal_encode = torch.cat([self.MAG(query, multimodal_memory).squeeze(0), encoder_state], dim=-1)


        attended_query, attented_wts = self.multimodal_attend(
            query, multimodal_memory, multimodal_memory
        )
        #multimodal_encode = torch.cat([attended_query.squeeze(0), encoder_state], dim=-1)


        return multimodal_encode


    


    def _setup_category_states(self):
        """Setup category states (focus + memory images).
        """
        # NOTE: Assumes three memory images; make it adaptive later.
        self.category_state = torch.stack(
            [
                self.category_embeds["focus"],
                self.category_embeds["memory"],
                self.category_embeds["memory"],
                self.category_embeds["memory"],
            ],
            dim=0,
        ).unsqueeze(0)


class Multimodal_Adaptation_Gate(nn.Module):
    def __init__(self, params):
        super(Multimodal_Adaptation_Gate, self).__init__()
        if params["text_encoder"] == "lstm":
            output_size = params["hidden_size"]
        else:
            output_size = params["word_embed_size"]
        self.gating_linear = nn.Linear(output_size*2, output_size)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.ones = torch.ones(1)
        self.ones = self.ones.cuda()

    def forward(self, query, metadata_states):
        query_m = query
        metadata_states_m = metadata_states.mean(dim=0)
        metadata_states_m = metadata_states_m.unsqueeze(0)
        g_v = self.gating_linear(torch.cat([query_m, metadata_states_m], dim=2))
        g_v = torch.tanh(g_v)
        H = self.linear(metadata_states_m)
        H = g_v.mul(H)
        beta = 0.75
        norms = torch.norm(query)/torch.norm(H)*beta
        alpha = torch.min(norms, self.ones)
        H = alpha*H
        Z = query.add(H)
        Z = self.layer_norm(Z)
        Z = self.dropout(Z)
        return Z
