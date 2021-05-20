import torch
import torch.nn as nn



class LSTM(nn.Module):

    

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

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
        out = self.dropout(out)

        return out



