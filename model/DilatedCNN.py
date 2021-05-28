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
    def __init__(self, input: pd.DataFrame, target_index: int, hyperparams: dict, horizon: int):

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
    def __init__(self, train_df: pd.DataFrame, oos_df:pd.DataFrame, target_index: int, hyperparams: dict, horizon: int):
        
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

class DilatedCNNWrapper:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        self.net = None
        self.target_index = None
        self.train_val = None
        self.test = None
        self.losses = None
        self.horizon = None
        self.period = None
        self.mae_rmse_ignore_when_actual_and_pred_are_zero = None
        self.mape_ignore_when_actual_is_zero = None
        self.cross_validation_objective = None
        self.cross_validation_objective_less_is_better = None
        self.cross_validation_results = None
        self.best_hyperparams = None
        self.best_mean_train_metrics = None
        self.best_mean_val_metrics = None
        self.trials = None
        self.train_val_metrics = None
        self.mean_test_metrics = None
        self.max_evals = None
        self.runtime_in_minutes = None

    def predict(self, input: pd.DataFrame, target_index: int, hps: dict, horizon: int) -> np.array:
        # When predicting on val set, input = train
        # When predicting on test set, input = train_val or val
        dataset = PredDataset(input, target_index, hps, horizon)
        pred_loader = DataLoader(dataset, batch_size=1, num_workers=1)
        input = next(iter(pred_loader))
        input = input.to(device = self.device)
        self.net.eval()
        self.net = self.net.to(device=self.device)
        pred = self.net(input.float())
        pred = pred.cpu().detach().numpy()
        pred = pred[0, 0, :]
        pred[pred < 0] = 0
        pred = pred.round()

        return pred

    def train(self, train_df: pd.DataFrame, oos_df: pd.DataFrame, target_index: int, hyperparams: dict, horizon: int):
        tic = time.time()
        in_channels = train_df.shape[1] if hyperparams['conditional'] is True else 1
        self.net = WaveNet(hyperparams, in_channels).to(device=self.device)
        self.net.train()
        self.losses = []

        train_dataset = TrainDataset(train_df = train_df, target_index = target_index, hyperparams=hyperparams, horizon=horizon)
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)
        oos_dataset = EvalDataset(train_df=train_df, oos_df=oos_df, target_index=target_index, hyperparams=hyperparams, horizon=horizon)
        oos_loader = DataLoader(oos_dataset, batch_size=1, num_workers=1)

        # define the loss and optimizer
        loss_fn = nn.L1Loss()
        optimizer = optim.Adam(self.net.parameters(), lr=hyperparams['learning_rate'])

        # training loop:
        inputs, labels = next(iter(train_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        oos_inputs, oos_labels = next(iter(oos_loader))
        oos_inputs, oos_labels = oos_inputs.to(self.device), oos_labels.to(self.device)

        best_oos_mae = None
        early_stopping = 0
        for epoch in range(hyperparams['max_epochs']):
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = self.net(inputs.float())
            loss = loss_fn(outputs, labels.float())
            self.losses.append(loss)

            oos_outputs = self.net(oos_inputs.float())
            oos_outputs[oos_outputs < 0] = 0
            oos_outputs = torch.round(oos_outputs)
            oos_mae = nn.L1Loss()(oos_labels.float(), oos_outputs.float())
        
            if best_oos_mae is None:
                best_oos_mae = oos_mae
                torch.save(self.net.state_dict(), 'DilatedCNN_checkpoint.pt')
            elif oos_mae < best_oos_mae and epoch > 20:
                best_oos_mae = oos_mae
                torch.save(self.net.state_dict(), 'DilatedCNN_checkpoint.pt')
                early_stopping = 0
            else:
                early_stopping += 1
            
            if early_stopping > hyperparams['early_stopping_rounds']:
                break

            reg_loss = np.sum([weights.norm(2) for weights in self.net.parameters()])

            total_loss = loss + hyperparams['l2_reg'] / 2 * reg_loss
            total_loss.backward()
            optimizer.step()

            # print statistics
            outputs[outputs < 0] = 0
            outputs = torch.round(outputs)
            train_mae = nn.L1Loss()(labels.float(), outputs.float())

            print('Epoch {} total loss: {} train mae: {} oos mae: {} best oos mae: {}'.format(epoch + 1, total_loss, train_mae, oos_mae, best_oos_mae))
        
        self.net = DilatedCNN(hyperparams, in_channels).to(device=self.device)
        self.net.load_state_dict(torch.load('DilatedCNN_checkpoint.pt'))
        self.net.eval()
        os.remove('DilatedCNN_checkpoint.pt')
        toc = time.time()
        print('Training time: {:.2f} seconds'.format(str(toc - tic)))    
    

