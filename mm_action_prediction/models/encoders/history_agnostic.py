"""Implements seq2seq encoder that is history-agnostic.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torchtext
from tools import rnn_support as rnn
from tools import torch_support as support
import models
# import models.encoders as encoders
from models import encoders
import gensim
import spacy
@encoders.register_encoder("history_agnostic")
class HistoryAgnosticEncoder(nn.Module):
    def __init__(self, params):
        super(HistoryAgnosticEncoder, self).__init__()
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
        if params["text_encoder"] == "transformer":
            layer = nn.TransformerEncoderLayer(
                params["word_embed_size"],
                params["num_heads_transformer"],
                params["hidden_size_transformer"],
            )
            self.encoder_unit = nn.TransformerEncoder(
                layer, params["num_layers_transformer"]
            )
            self.pos_encoder = models.PositionalEncoding(params["word_embed_size"])
        elif params["text_encoder"] == "lstm":
            self.encoder_unit = nn.LSTM(
                encoder_input_size,
                params["hidden_size"],
                params["num_layers"],
                batch_first=True,
            )
        else:
            raise NotImplementedError("Text encoder must be transformer or LSTM!")

    def forward(self, batch):
        """Forward pass through the encoder.

        Args:
            batch: Dict of batch variables.

        Returns:
            encoder_outputs: Dict of outputs from the forward pass.
        """
        encoder_out = {}
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Flatten for history_agnostic encoder.
        batch_size, num_rounds, max_length = batch["user_utt"].shape
        encoder_in = support.flatten(batch["user_utt"], batch_size, num_rounds)
        encoder_len = support.flatten(batch["user_utt_len"], batch_size, num_rounds)
        if self.params["embedding_type"]=="random":
            word_embeds_enc = self.word_embed_net(encoder_in)
        elif self.params["embedding_type"]=="glove":
            word_embeds_enc = torch.tensor([[self.nlp(batch["ind2word"][int(encoder_in[row][col])]).vector for col in range(encoder_in.shape[1])] for row in range(encoder_in.shape[0])], requires_grad=True).to(device)
        elif self.params["embedding_type"]=="word2vec":
            word_embeds_enc = torch.stack([torch.stack([self.word_to_vec(encoder_in, row, col, batch["ind2word"]) for col in range(encoder_in.shape[1])]) for row in range(encoder_in.shape[0])])
            word_embeds_enc.requires_grad_(requires_grad=True)
        elif self.params["embedding_type"]=="fasttext":
            word_list = [[batch["ind2word"][int(encoder_in[row][col])] for col in range(encoder_in.shape[1])] for row in range(encoder_in.shape[0])]
            word_embeds_enc = torch.stack([self.fasttext_model.get_vecs_by_tokens(row) for row in word_list]).to(device)
            word_embeds_enc.requires_grad=True
        # Text encoder: LSTM or Transformer.
        if self.params["text_encoder"] == "lstm":
            all_enc_states, enc_states = rnn.dynamic_rnn(
                self.encoder_unit, word_embeds_enc, encoder_len, return_states=True
            )
            encoder_out["hidden_states_all"] = all_enc_states
            encoder_out["hidden_state"] = enc_states

        elif self.params["text_encoder"] == "transformer":
            enc_embeds = self.pos_encoder(word_embeds_enc).transpose(0, 1)
            enc_pad_mask = batch["user_utt"] == batch["pad_token"]
            enc_pad_mask = support.flatten(enc_pad_mask, batch_size, num_rounds)
            enc_states = self.encoder_unit(
                enc_embeds, src_key_padding_mask=enc_pad_mask
            )
            encoder_out["hidden_states_all"] = enc_states.transpose(0, 1)
        return encoder_out
    
    def word_to_vec(self, encoder_in, row, col, ind2word):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            return torch.from_numpy(self.w2v_model[ind2word[int(encoder_in[row][col])]]).requires_grad_(requires_grad=True).to(device)
        except KeyError as k:
            return torch.zeros(300, requires_grad=True).to(device)
