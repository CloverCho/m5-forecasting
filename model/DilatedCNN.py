# https://github.com/litanli/wavenet-time-series-forecasting/blob/master/wavenet_pytorch.py
import os
import time

import torch
import torch.nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd



class DilatedCausalConv1d(nn.Module):
    def __init__(self, hyperparams: dict, dilation_factor: int, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)
        
        self.dilation_factor = dilation_factor
        self.dilated_causal_conv = nn.Conv1d(in_channels=in_channels, 
                                             out_channels=hyperparams['nb_filters'],
                                             kernel_size=hyperparams['kernel_size']
                                             dilation=dilation_factor)
        self.dilated_causal_conv.apply(weights_init)

        self.skip_connection = nn.Conv1d(in_channels=in_channels, out_channels=hyperparams['nb_filters'], kernel_size=1)
        self.skip_connection.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.leaky_relu(self.dilated_causal_conv(x))
        x2 = x[:, :, self.dilation_factor:]
        x2 = self.skip_connection(x2)
        return x1 + x2
    
class PredDataset(Dataset):
    def __init__(self, input, target_index, hyperparams: dict, horizon: int):

        input = input.values
        receptive_field = 2 ** (hyperparams['nb_layers'] - 1) * hyperparams['kernel_size']
        padding = receptive_field - 1

        if hyperparams['conditional']:
            seq_x = input[-horizon:, :]
        else:
            seq_x = input[-horizon:, target_index].reshape(-1, 1)

        # left-zero-pad inputs in the timesteps dimension
        seq_x = np.pad(seq_x, pad_width=((padding, 0), (0, 0)), mode='constant')

        seq_x = seq_x.T
        self.seq_x = seq_x

    def __len__(self):

        return self.seq_x.shape[1]
    def __getitem__(self, idx):
        return self.seq_x


class EvalDataset(Dataset):
    def __init__(self, train_df, oos_df, target_index, hyperparams: dict, horizon: int):
        
        train_df = train_df.values
        oos_df = oos_df.values
        receptive_field = 2 ** (hyperparams['nb_layers'] - 1) * hyperparams['kernel_size']
        padding = receptive_field - 1

        if hyperparams['conditional']:
            seq_x = train_df[-horizon:, :]
            seq_y = oos_df[:horizon, target_index].reshape(-1, 1)
        else:
            seq_x = train_df[-horizon:, target_index].reshape(-1, 1)
            seq_y = oos_df[:horizon, target_index].reshape(-1, 1)

        # left-zero-pad inputs in the timesteps dimension
        seq_x = np.pad(seq_x, pad_width=((padding, 0), (0, 0)), mode='constant')

        seq_x = seq_x.T
        seq_y = seq_y.T

        self.seq_x = seq_x
        self.seq_y = seq_y

    def __len__(self):
        return self.seq_x.shape[1]

    def __getitem__(self, idx):
        return self.seq_x, self.seq_y


class DilatedCNN(nn.Module):
    def __init__(self, hyperparams: dict, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)
        
        self.dilation_factors = [2 ** i for i in range(0, hyperparams['nb_layers'])]
        self.in_channels = [in_channels] + [hyperparams['nb_filters'] for _ in range(hyperparams['nb_layers'])]
        self.dilated_causal_convs = nn.ModuleList(
            [DilatedCausalConv1d(hyperparams, self.dilation_factors[i], self.in_channels[i]) for i in range(hyperparams['nb_layers'])]
        )
        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.output_layer = nn.Conv1d(in_channels=self.in_channels[-1], out_channels = 1, kernel_size=1)
        self.output_layer.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        for dilated_causal_conv in self.dilated_causal_convs:
            x = dilated_causal_conv(x)
        out = self.leaky_relu(self.output_layer(x))



class CustomConv1d(nn.Module):
    def __init__(self, in_channels, dilation_factor, kernel_size, nb_filters):
        pass