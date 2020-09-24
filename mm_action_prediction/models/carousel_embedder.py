"""Embedding carousel for action predictions.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn


class CarouselEmbedder(nn.Module):
    def __init__(self, params):
        super(CarouselEmbedder, self).__init__()
        self.params = params
        self.host = torch.cuda if params["use_gpu"] else torch
        self.positions = ["left", "center", "right", "focus", "empty"]
        self.occupancy_states = {}
        self.carousel_pos = {}
        for position in self.positions:
            pos_parameter = torch.randn(params["word_embed_size"])
            if params["use_gpu"]:
                pos_parameter = pos_parameter.cuda()
            pos_parameter = nn.Parameter(pos_parameter)
            self.carousel_pos[position] = pos_parameter
            # Register the parameter for training/saving.
            self.register_parameter(position, pos_parameter)

        # Project carousel embedding to same size as encoder.
        input_size = params["asset_feature_size"] + params["word_embed_size"]
        if params["text_encoder"] == "lstm":
            output_size = params["hidden_size"]
        else:
            output_size = params["word_embed_size"]
        self.carousel_embed_net = nn.Linear(input_size, output_size)
        self.carousel_attend = nn.MultiheadAttention(output_size, 1)
        self.carousel_mask = self._generate_carousel_mask(3)
        #query mask add
        self.query_mask = self._generate_carousel_mask(1)

        self.MAG = Multimodal_Adaptation_Gate(params)
        self.MAG_only = Multimodal_Adaptation_Gate_only(params)
        self.Visualgate = VisualGate(params)

    def forward(self, carousel_state, encoder_state, encoder_size):
        """Carousel Embedding.

        Args:
            carousel_state: State of the carousel
            encoder_state: State of the encoder
            encoder_size: (batch_size, num_rounds)

        Returns:
            new_encoder_state:
        """
        if len(self.occupancy_states) == 0:
            self._setup_occupancy_states()

        batch_size, num_rounds = encoder_size
        carousel_states = []
        carousel_sizes = []
        for inst_id in range(batch_size):
            for round_id in range(num_rounds):
                round_datum = carousel_state[inst_id][round_id]
                if round_datum is None:
                    carousel_features = self.none_features
                    carousel_sizes.append(1)
                elif "focus" in round_datum:
                    carousel_features = torch.cat(
                        [round_datum["focus"], self.carousel_pos["focus"]]
                    ).unsqueeze(0)
                    carousel_features = torch.cat(
                        [carousel_features, self.empty_feature, self.empty_feature],
                        dim=0,
                    )
                    carousel_sizes.append(1)
                elif "carousel" in round_datum:
                    carousel_size = len(round_datum["carousel"])
                    if carousel_size < 3:
                        all_embeds = torch.cat(
                            [round_datum["carousel"]]
                            + self.occupancy_embeds[carousel_size],
                            dim=0,
                        )
                    else:
                        all_embeds = round_datum["carousel"]
                    all_states = self.occupancy_states[carousel_size]
                    carousel_features = torch.cat([all_embeds, all_states], -1)
                    carousel_sizes.append(carousel_size)
                # Project into same feature shape.
                carousel_features = self.carousel_embed_net(carousel_features)
                carousel_states.append(carousel_features)
        # Shape: (L,N,E)
        carousel_states = torch.stack(carousel_states, dim=1)
        # Mask: (N,S)
        carousel_len = self.host.LongTensor(carousel_sizes)
        query_len = torch.ones(len(carousel_sizes), dtype=torch.long)
        query_len = query_len.cuda()
        query = encoder_state.unsqueeze(0)

        ###using gate for carousel, query
        #MAG+encoder_state
        
        if self.params['gate_type'] == "MAG":
            carousel_encode = torch.cat([self.MAG(query, carousel_states).squeeze(0), encoder_state], dim=-1)
        elif self.params['gate_type'] == "none":
            ##original
            attended_query, attended_wts = self.carousel_attend(
                query,
                carousel_states,
                carousel_states,
                key_padding_mask = self.carousel_mask[carousel_len-1]
            )
            carousel_encode = torch.cat([attended_query.squeeze(0), encoder_state], dim=-1)
        elif self.params['gate_type'] == "MMI":
            #inkoon gate
            attended_query_q, attended_wts_q = self.carousel_attend(
                query,
                carousel_states,
                carousel_states,
                key_padding_mask=self.carousel_mask[carousel_len-1],
            )
            v_input = carousel_states.mean(dim=0)
            v_input = v_input.unsqueeze(0)
            attended_query_p, atteded_wts_p = self.carousel_attend(
                v_input,
                query,
                query,
                key_padding_mask=self.query_mask[query_len -1],
            )
            attended_query_a, attended_wts_a = self.carousel_attend(
                query,
                attended_query_p,
                attended_query_p,
                key_padding_mask = self.query_mask[query_len-1],
            )
            b = self.Visualgate(attended_query_a, attended_query_q)
            carousel_encode = torch.cat([attended_query_a.squeeze(0), b.squeeze(0)], dim=-1)
            ############end

        return carousel_encode

    def empty_carousel(self, carousel_state):
        """Check if carousel is empty in the standard representation.

        Args:
            carousel_state: Carousel state

        Returns:
            empty_carousel: Boolean (True -- empty, False -- not empty)
        """
        return carousel_state == {"focus": None, "carousel": []}

    def _generate_carousel_mask(self, size):
        """Generates square masks for transformers to avoid peeking.
        """
        mask = (torch.triu(torch.ones(size, size)) == 0).transpose(0, 1)
        if self.params["use_gpu"]:
            mask = mask.cuda()
        return mask

    def _setup_occupancy_states(self):
        """Setup carousel states and embeddings for different occupancy levels.
        """
        self.occupancy_states = {}
        self.occupancy_embeds = {}
        self.zero_tensor = self.host.FloatTensor(self.params["asset_feature_size"])
        self.zero_tensor.fill_(0.0)
        for num_items in range(4):
            states = [self.carousel_pos[ii] for ii in self.positions[:num_items]]
            states += [self.carousel_pos["empty"] for ii in range(3 - num_items)]
            states = torch.stack(states, dim=0)
            self.occupancy_states[num_items] = states
            embeds = [self.zero_tensor for _ in range(3 - num_items)]
            if len(embeds):
                embeds = [torch.stack(embeds, dim=0)]
            self.occupancy_embeds[num_items] = embeds
        self.empty_feature = torch.cat(
            [self.zero_tensor, self.carousel_pos["empty"]], dim=-1
        ).unsqueeze(0)
        self.none_features = self.empty_feature.expand(3, -1)


class VisualGate(nn.Module):
    def __init__(self, params):
        super(VisualGate, self).__init__()
        if params["text_encoder"] == "lstm":
            output_size = params["hidden_size"]
        else:
            output_size = params["word_embed_size"]
        self.a_linear = nn.Linear(output_size, output_size)
        self.q_linear = nn.Linear(output_size, output_size)
    def forward(self, a, q):
        wa = self.a_linear(a)
        wq = self.q_linear(q)
        wa_wq = wa + wq
        g = torch.sigmoid(wa_wq)
        return g


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
        query_m = query    #(1,200,512)
        metadata_states_m = metadata_states.mean(dim=0)  #(200, 512)
        metadata_states_m = metadata_states_m.unsqueeze(0)  #(1, 200, 512)
        g_v = self.gating_linear(torch.cat([query_m, metadata_states_m], dim=2)) #(1, 200, 1024)->(1, 200, 512)
        g_v = torch.tanh(g_v)   #(1, 200, 512)
        H = self.linear(metadata_states_m) #(1, 1, 512)
        H = g_v.mul(H)    #(1, 200, 512)
        beta = 0.75
        norms = torch.norm(query)/torch.norm(H)*beta
        alpha = torch.min(norms, self.ones)
        H = alpha*H
        Z = query.add(H) #(1, 200, 512)
        Z = self.layer_norm(Z)
        Z = self.dropout(Z)  #(1, 200, 512)
        
        return Z

class Multimodal_Adaptation_Gate_only(nn.Module):
    def __init__(self, params):
        super(Multimodal_Adaptation_Gate_only, self).__init__()
        if params["text_encoder"] == "lstm":
            output_size = params["hidden_size"]
        else:
            output_size = params["word_embed_size"]
        self.gating_linear = nn.Linear(output_size*2, output_size*2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(output_size, output_size*2)
        self.linear_q = nn.Linear(output_size, output_size*2)
        self.layer_norm = nn.LayerNorm(output_size*2)
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
        beta = 0.5
        query_l = self.linear_q(query)
        norms = torch.norm(query_l)/torch.norm(H)*beta
        alpha = torch.min(norms, self.ones)
        H = alpha*H
        Z = query_l.add(H)
        Z = self.layer_norm(Z)
        Z = self.dropout(Z)
        return Z
