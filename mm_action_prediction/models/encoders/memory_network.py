"""Implements memory network encoder.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torchtext
from tools import rnn_support as rnn
from tools import torch_support as support
# import models.encoders as encoders
from models import encoders
import gensim
import spacy
@encoders.register_encoder("memory_network")
class MemoryNetworkEncoder(nn.Module):
    def __init__(self, params):
        super(MemoryNetworkEncoder, self).__init__()
        self.params = params

        self.word_embed_net = nn.Embedding(
            params["vocab_size"], params["word_embed_size"]
        )
        encoder_input_size = params["word_embed_size"]
        self.encoder_input_size = encoder_input_size
        if self.params["embedding_type"]=="glove":
            self.nlp = spacy.load("en_vectors_web_lg")
        elif self.params["embedding_type"]=="word2vec":
            self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/home/yeonseok/GoogleNews-vectors-negative300.bin', binary=True)
        elif self.params["embedding_type"]=="fasttext":
            self.fasttext_model = torchtext.vocab.FastText('en')
        self.encoder_unit = nn.LSTM(
            encoder_input_size,
            params["hidden_size"],
            params["num_layers"],
            batch_first=True,
        )
        self.fact_unit = nn.LSTM(
            params["word_embed_size"],
            params["hidden_size"],
            params["num_layers"],
            batch_first=True,
        )

        self.softmax = nn.functional.softmax
        self.fact_attention_net = nn.Sequential(
            nn.Linear(2 * params["hidden_size"], params["hidden_size"]),
            nn.ReLU(),
            nn.Linear(params["hidden_size"], 1),
        )

    def forward(self, batch):
        """Forward pass through the encoder.

        Args:
            batch: Dict of batch variables.

        Returns:
            encoder_outputs: Dict of outputs from the forward pass.
        """
        encoder_out = {}
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Flatten to encode sentences.
        batch_size, num_rounds, _ = batch["user_utt"].shape
        encoder_in = support.flatten(batch["user_utt"], batch_size, num_rounds)
        encoder_len = batch["user_utt_len"].reshape(-1)
        if self.params["embedding_type"]=="random":
            word_embeds_enc = self.word_embed_net(encoder_in)
        elif self.params["embedding_type"]=="glove":
            word_embeds_enc = torch.tensor([[self.nlp(batch["ind2word"][int(encoder_in[row][col])]).vector for col in range(encoder_in.shape[1])] for row in range(encoder_in.shape[0])], requires_grad=True).to(torch.device("cuda:0"))
        elif self.params["embedding_type"]=="word2vec":
            #word_embeds_enc = torch.zeros(encoder_in.shape[0], encoder_in.shape[1], self.encoder_input_size, requires_grad=True).to(device)
            #for row in range(encoder_in.shape[0]):
                #for col in range(encoder_in.shape[1]):
                    #try:
                        #word_embeds_enc[row][col] = torch.from_numpy(self.w2v_model[batch["ind2word"][int(encoder_in[row][col])]]).requires_grad_(requires_grad=True).to(device)
                    #except KeyError as k:
                        #word_embeds_enc[row][col] = torch.zeros(300, requires_grad=True).to(device)
            word_embeds_enc = torch.stack([torch.stack([self.word_to_vec(encoder_in, row, col, batch["ind2word"]) for col in range(encoder_in.shape[1])]) for row in range(encoder_in.shape[0])])
            word_embeds_enc.requires_grad_(requires_grad=True)            
        elif self.params["embedding_type"]=="fasttext":
            word_list = [[batch["ind2word"][int(encoder_in[row][col])] for col in range(encoder_in.shape[1])] for row in range(encoder_in.shape[0])]
            word_embeds_enc = torch.stack([self.fasttext_model.get_vecs_by_tokens(row) for row in word_list]).to(device)
            word_embeds_enc.requires_grad=True
        # Fake encoder_len to be non-zero even for utterances out of dialog.
        fake_encoder_len = encoder_len.eq(0).long() + encoder_len
        all_enc_states, enc_states = rnn.dynamic_rnn(
            self.encoder_unit, word_embeds_enc, fake_encoder_len, return_states=True
        )
        encoder_out["hidden_states_all"] = all_enc_states
        encoder_out["hidden_state"] = enc_states

        utterance_enc = enc_states[0][-1]
        batch["utterance_enc"] = support.unflatten(
            utterance_enc, batch_size, num_rounds
        )
        encoder_out["dialog_context"] = self._memory_net_forward(batch)
        return encoder_out

    def word_to_vec(self, encoder_in, row, col, ind2word):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            return torch.from_numpy(self.w2v_model[ind2word[int(encoder_in[row][col])]]).requires_grad_(requires_grad=True).to(device)
        except KeyError as k:
            return torch.zeros(300, requires_grad=True).to(device)

    def _memory_net_forward(self, batch):
        """Forward pass for memory network to look up fact.

        1. Encodes fact via fact rnn.
        2. Computes attention with fact and utterance encoding.
        3. Attended fact vector and question encoding -> new encoding.

        Args:
          batch: Dict of hist, hist_len, hidden_state
        """
        # kwon : fact = prevuiys utterance + response concatenated as one
        # For example, 'What is the color of the couch? A : Red.'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size, num_rounds, enc_time_steps = batch["fact"].shape
        all_ones = np.full((num_rounds, num_rounds), 1)
        fact_mask = np.triu(all_ones, 1)
        fact_mask = np.expand_dims(np.expand_dims(fact_mask, -1), 0)
        fact_mask = torch.BoolTensor(fact_mask)
        if self.params["use_gpu"]:
            fact_mask = fact_mask.cuda()
        fact_mask.requires_grad_(False)

        fact_in = support.flatten(batch["fact"], batch_size, num_rounds)
        fact_len = support.flatten(batch["fact_len"], batch_size, num_rounds)
        if self.params["embedding_type"]=="random":
            fact_embeds  = self.word_embed_net(fact_in)
        elif self.params["embedding_type"]=="glove":
            fact_embeds  = torch.tensor([[self.nlp(batch["ind2word"][int(fact_in[row][col])]).vector for col in range(fact_in.shape[1])] for row in range(fact_in.shape[0])], requires_grad=True).to(device)
        elif self.params["embedding_type"]=="word2vec":
            fact_embeds  = torch.stack([torch.stack([self.word_to_vec(fact_in, row, col, batch["ind2word"]) for col in range(fact_in.shape[1])]) for row in range(fact_in.shape[0])])
            fact_embeds .requires_grad_(requires_grad=True)
        elif self.params["embedding_type"]=="fasttext":
            word_list = [[batch["ind2word"][int(fact_in[row][col])] for col in range(fact_in.shape[1])] for row in range(fact_in.shape[0])]
            fact_embeds  = torch.stack([self.fasttext_model.get_vecs_by_tokens(row) for row in word_list]).to(device)
            fact_embeds.requires_grad=True
        # Encoder fact and unflatten the last hidden state.
        _, (hidden_state, _) = rnn.dynamic_rnn(
            self.fact_unit, fact_embeds, fact_len, return_states=True
        )
        fact_encode = support.unflatten(hidden_state[-1], batch_size, num_rounds)
        fact_encode = fact_encode.unsqueeze(1).expand(-1, num_rounds, -1, -1)

        utterance_enc = batch["utterance_enc"].unsqueeze(2)
        utterance_enc = utterance_enc.expand(-1, -1, num_rounds, -1)
        # Combine, compute attention, mask, and weight the fact encodings.
        combined_encode = torch.cat([utterance_enc, fact_encode], dim=-1)
        attention = self.fact_attention_net(combined_encode)
        attention.masked_fill_(fact_mask, float("-Inf"))
        attention = self.softmax(attention, dim=2)
        attended_fact = (attention * fact_encode).sum(2)
        return attended_fact
