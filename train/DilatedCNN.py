import torch
import numpy as np
import pandas as pd

from model.DilatedCNN import *
from model.losses import RMELoss
from fastprogress import master_bar, progress_bar



class DilatedCNN_Train:
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

    def predict(self, input, target_index, hps: dict, horizon: int) -> np.array:
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

    def train(self, train_df, oos_df, target_index,  horizon, num_epochs, lr, nb_layers, nb_filters):
        tic = time.time()

        hyperparams = {'max_epochs': num_epochs,
                       'learning_rate': lr,
                       'conditional': True ,
                       'nb_layers': nb_layers,
                       'nb_filters': nb_filters}   
   
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
    


