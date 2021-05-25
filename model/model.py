import torch
import torch.nn as nn
from torch.autograd import Variable

class SingleRNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(SingleRNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = 0.2)
        
        self.rnn = nn.RNN(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        ula, h_out = self.rnn(x, h_0)

        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        out = self.dropout(out)

        return out


class singleLSTM(nn.Module):

    

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(singleLSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = 0.2)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        

        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        #out = self.dropout(out)

        return out

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.batch_size = 1
        # self.seq_length = seq_length

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                             dropout=0.2)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(256, 128)

        self.bn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device))

        _, (hn, cn) = self.LSTM(x, (h_1, c_1))

        print("hidden state shape is:",hn.size())
        y = hn.view(-1, self.hidden_size)

        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        print("final state shape is:",final_state.shape)

        x0 = self.fc1(final_state)
        x0 = self.bn1(x0)
        x0 = self.dp1(x0)
        x0 = self.relu(x0)

        x0 = self.fc2(x0)
        x0 = self.bn2(x0)
        x0 = self.dp2(x0)

        x0 = self.relu(x0)

        out = self.fc3(x0)
        print(out.size())
        return out
    

class SingleGRU(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(SingleGRU,self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = 0.2)

        self.gru = nn.GRU(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)

        self.fc - nn.Linear(hidden_size, num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        ula, h_out = self.gru(x, h_0)

        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        out = self.dropout(out)

        return out


class GRU(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.batch_size = 1
        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.2)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(256, 128)

        self.bn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device))


        _, hn = self.GRU(x, h_1)

        print("hidden state shape is:",hn.size())
        y = hn.view(-1, self.hidden_size)

        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        print("final state shape is:",final_state.shape)

        x0 = self.fc1(final_state)
        x0 = self.bn1(x0)
        x0 = self.dp1(x0)
        x0 = self.relu(x0)

        x0 = self.fc2(x0)
        x0 = self.bn2(x0)
        x0 = self.dp2(x0)

        x0 = self.relu(x0)

        out = self.fc3(x0)
        print(out.size())
        return out
    
# class Encoder(nn.Module):
