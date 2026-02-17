#!/usr/bin/env python3

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import model_data 

# Set seed for reproducibility.
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # For deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Positional encoding for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
# CNN layers for multi-scale overview
class OverviewCNN(nn.Module):
    def __init__(self, seq_input_dimension, kernal=1):
        super(OverviewCNN, self).__init__()
        self.kernal = kernal
        self.conv_layer = nn.Conv1d(seq_input_dimension, 8, kernel_size=kernal, padding="same")
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.activation(x)
        return x


# Transformer-based sequence-to-sequence model
class TransformerSeq2Seq(nn.Module):
    def __init__(self,
                 output_len=199,
                 output_dim=1,
                 max_seq_length=200,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1):
        super(TransformerSeq2Seq, self).__init__()
        # set seed for reproducibility
        seed = 42
        set_seed(seed)

        self.d_model = d_model

        self.output_len = output_len
        self.output_dim = output_dim

        self.max_seq_length = max_seq_length
        self.seq_input_length = model_data.get_seq_encoding_length(self.max_seq_length)
        self.seq_input_dimension = model_data.get_seq_encoding_dimension()
        
        # Meta processing
        self.meta_input_length = model_data.get_meta_length()
        self.meta_dimension = model_data.get_meta_embedding_dimension()
        self.meta_activation = nn.ReLU()
        self.meta_dense = nn.Linear(self.meta_input_length, self.meta_dimension)
        self.repeat_length = self.seq_input_length

        self.global_embed_dimension = 8 * 8  # from multi-scale convs

        self.total_embed_dimension = self.seq_input_dimension + self.meta_dimension + self.global_embed_dimension   # 8 + 64 = 72

        # Overview convolutions (kernels 1-8) for global features
        conv_list = []
        for i in range(1,9):
            conv = OverviewCNN(self.seq_input_dimension, kernal=i)
            conv_list.append(conv)
        self.overview_convs = nn.ModuleList(conv_list)

        self.conv_activation = nn.ReLU()

        # Input projection
        self.input_projection = nn.Linear(self.total_embed_dimension, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_input_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, 
                                                         num_layers=num_decoder_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, self.output_dim)

        # Learnable target embeddings for decoder input
        self.tgt_embedding = nn.Parameter(torch.randn(1, self.seq_input_length, d_model))

        self.sigmoid = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, seq_encoding_input, meta_input, mask_input, charge_mask_input):
        batch_size = seq_encoding_input.size(0)
        # Process meta information
        info = self.meta_dense(meta_input)
        info = self.meta_activation(info)
        info = info.unsqueeze(1)  # info shape:(batch_size, 1, 8)

        overview = []
        # (batch_size, channels, seq_input_len)
        encoding_transposed = seq_encoding_input.transpose(1, 2)  
        for i, conv in enumerate(self.overview_convs):
            x = conv(encoding_transposed)  # (batch_size, 8, seq_input_len)
            overview.append(x)  
        overview = torch.cat(overview, dim=1)  # (batch_size, 64, seq_input_len)
        overview = overview.transpose(1, 2)  # (batch_size, seq_input_len, 64)
        #print("overview shape:", overview.shape)

        # repeat along dimension 1
        info = info.repeat(1, self.repeat_length, 1) # (batch_size, seq_input_len, 8)
        info = torch.cat([info, overview], dim=-1)  # (batch_size, seq_input_len, 72)
        repeat_mask = mask_input.unsqueeze(2) # (batch_size, seq_input_len, 1)
        info = info * repeat_mask  # apply mask to info

        # Concatenate embedding and meta info
        x = torch.cat([seq_encoding_input, info], dim=-1) # (batch_size, seq_input_len, 99)

        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch_size, seq_input_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encoder
        memory = self.transformer_encoder(x)  # (batch_size, seq_input_len, d_model)

        # Prepare decoder input 
        tgt = self.tgt_embedding.expand(batch_size, -1, -1)  # (batch_size, seq_input_len, d_model)
        tgt = self.pos_encoder(tgt)

        # Decoder
        output = self.transformer_decoder(tgt, memory)  # (batch_size, seq_input_len, d_model)
        # Project to output dimension
        output = self.output_projection(output)  # (batch_size, seq_input_len, output_dim)

        output = self.sigmoid(output)
        mask = mask_input.unsqueeze(-1)  # (batch_size, seq_input_len, 1)
        # Apply training mask 
        output = output * mask  # mask 

        if self.output_dim > 2:
            #print("charge mask input", charge_mask_input[0,:10])
            charge_mask = charge_mask_input.unsqueeze(1)  # (batch_size, seq_input_len, 1)
            output = output * charge_mask  # mask
            #print(charge_mask[0,0,:])
            #for i in range(30):
            #    print(i, "mask", charge_mask[0,0,i])
            #    print("output", output[0,:,i])

        # Keep only the output 
        output = output[:, 1:self.output_len + 1,:]  
        #print("output.shape:", output.shape)
        #print(output[0,:,0])  # print first sample's output

        output = output.reshape(batch_size, self.output_len * self.output_dim)  # (batch_size, seq_input_len * output_dim)
        #print(output[0,:])  # print first sample's output after reshape
        #print("output.shape after reshape:", output.shape)
        return output


def build(**kwargs):
    return TransformerSeq2Seq(**kwargs)
