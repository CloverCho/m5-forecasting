import numpy as np
import pandas as pd
import torch
import torch.onnx

import hiddenlayer as hl
from torch.utils.tensorboard import SummaryWriter
from model import *


def main():
    transforms = [hl.transforms.Prune('Constant')]

    num_layers = 4
    hidden_size = 512

    lstm_input_size = 1262
    lstm_num_classes = 1262
    lstm_input = torch.Tensor(np.ones((20, 28, 1262), dtype=np.float))

    writer0 = SummaryWriter()
    model = singleLSTM(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1)
    writer0.add_graph(model, lstm_input)
    writer0.close()

    writer1 = SummaryWriter()
    model = LSTM(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size,
                 num_layers=num_layers)
    writer1.add_graph(model, lstm_input)
    writer1.close()

    writer2 = SummaryWriter()
    model = singleGRU(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size,
                      num_layers=num_layers)
    writer2.add_graph(model, lstm_input)
    writer2.close()

    writer3 = SummaryWriter()
    model = GRU(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size,
                num_layers=num_layers)
    writer3.add_graph(model, lstm_input)
    writer3.close()

    writer4 = SummaryWriter()
    model = singleRNN(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1)
    writer4.add_graph(model, lstm_input)
    writer4.close()
    
    writer5 = SummaryWriter()
    EDA_input_size = 512
    EDA_num_classes = 512
    encoder_input = torch.Tensor(np.ones((28, 512), dtype=np.float))
    decoder_input = torch.Tensor(np.ones((1, 512), dtype=np.float))

    model = Seq2Seq(28, num_classes=EDA_num_classes, input_size=EDA_input_size, embedding_dim = hidden_size)
    writer5.add_graph(model, (encoder_input, decoder_input))
    writer5.close()


if __name__ == "__main__":
    main()

    #tensorboard --logdir runs
