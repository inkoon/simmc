"""Implements hierarchical recurrent neural network encoder.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torchtext
from tools import rnn_support as rnn
from tools import torch_support as support
# import models.encoders as encoders
from models import encoders
import spacy
import gensim
@encoders.register_encoder("hierarchical_recurrent")
class HierarchicalRecurrentEncoder(nn.Module):
    def __init__(self, params):
        super(HierarchicalRecurrentEncoder, self).__init__()
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
        self.dialog_unit = nn.LSTM(
            params["hidden_size"],
            params["hidden_size"],
            params["num_layers"],
            batch_first=True,
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
            word_embeds_enc = torch.stack([torch.stack([self.word_to_vec(encoder_in, row, col, batch["ind2word"]) for col in range(encoder_in.shape[1])]) for row in range(encoder_in.shape[0])])
            word_embeds_enc.requires_grad_(requires_grad=True).to(device)
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
        new_size = (batch_size, num_rounds, utterance_enc.shape[-1])
        utterance_enc = utterance_enc.reshape(new_size)
        encoder_out["dialog_context"], _ = rnn.dynamic_rnn(
            self.dialog_unit, utterance_enc, batch["dialog_len"], return_states=True
        )
        return encoder_out

    def word_to_vec(self, encoder_in, row, col, ind2word):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            return torch.from_numpy(self.w2v_model[ind2word[int(encoder_in[row][col])]]).requires_grad_(requires_grad=True).to(device)
        except KeyError as k:
            return torch.zeros(300, requires_grad=True).to(device)
