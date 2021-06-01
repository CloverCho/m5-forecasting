import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class singleRNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(singleRNN, self).__init__()

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

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(hidden_size, 512)
        self.bn1 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device))

        _, (hn, cn) = self.LSTM(x, (h_1, c_1))
        print("hidden state shape is:",hn.size())

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

        return out
        
        
    

class singleGRU(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(singleGRU,self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = 0.2)

        self.gru = nn.GRU(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)

        self.fc = nn.Linear(hidden_size, num_classes)

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
        # self.seq_length = seq_length

        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.2)

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
        
        return out


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim
        self.num_layers = 3
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.35
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.device))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.device))

        x, (hidden, cell) = self.rnn1(x, (h_1, c_1))

        # return hidden_n.reshape((self.n_features, self.embedding_dim))
        return x, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden[2:3, :, :]

        # print("hidden size is",hidden.size())

        # repeat decoder hidden state src_len times
        # hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        hidden = hidden.repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # print("encode_outputs size after permute is:",encoder_outputs.size())

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=1,
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.35
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, input_hidden, input_cell):
        x = x.reshape((1, 1, 1))

        x, (hidden_n, cell_n) = self.rnn1(x, (input_hidden, input_cell))

        x = self.output_layer(x)
        return x, hidden_n, cell_n


class AttentionDecoder(nn.Module):
    def __init__(self, seq_len, attention, input_dim=64, n_features=1, encoder_hidden_state=512):
        super(AttentionDecoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.attention = attention

        self.rnn1 = nn.LSTM(
            # input_size=1,
            input_size=encoder_hidden_state + 1,  # Encoder Hidden State + One Previous input
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.35
        )

        self.output_layer = nn.Linear(self.hidden_dim * 2, n_features)

    def forward(self, x, input_hidden, input_cell, encoder_outputs):
        a = self.attention(input_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        x = x.reshape((1, 1, 1))

        rnn_input = torch.cat((x, weighted), dim=2)

        # x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))
        x, (hidden_n, cell_n) = self.rnn1(rnn_input, (input_hidden, input_cell))

        output = x.squeeze(0)
        weighted = weighted.squeeze(0)

        x = self.output_layer(torch.cat((output, weighted), dim=1))
        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64, output_length=28):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.attention = Attention(512, 512)
        self.output_length = output_length
        self.decoder = AttentionDecoder(seq_len, self.attention, embedding_dim, n_features).to(device)

    def forward(self, x, prev_y):
        encoder_output, hidden, cell = self.encoder(x)

        # Prepare place holder for decoder output
        targets_ta = []
        # prev_output become the next input to the LSTM cell
        prev_output = prev_y

        # itearate over LSTM - according to the required output days
        for out_days in range(self.output_length):
            prev_x, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell, encoder_output)
            hidden, cell = prev_hidden, prev_cell
            prev_output = prev_x

            targets_ta.append(prev_x.reshape(1))

        targets = torch.stack(targets_ta)

        return targets




class DilatedCNN(nn.Module):
    # https://vaibhaw-vipul.medium.com/building-a-dilated-convnet-in-pytorch-f7c1496d9bf5
    def __init__(self):
        super(DilatedCNN, self).__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=9, stride=1, padding=0, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.ReLU(),
        )

        self.fclayers = nn.Sequential(
            nn.Linear(2304, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.convlayers(x)
        x = x.view(-1, 2304)
        out = self.fclayers(x)
        return out
