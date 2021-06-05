import numpy as np
import pandas as pd
import torch
import torch.onnx

from torchviz import make_dot
from torchsummaryX import summary
import hiddenlayer as hl

from model import *


def main():
    transforms = [hl.transforms.Prune('Constant')]

    num_layers = 4
    hidden_size = 512
    
    lstm_input_size = 1262
    lstm_num_classes = 1262
    #lstm_input = pd.DataFrame(np.ones((1262, 28, 3374), dtype=np.float))
    inputs = torch.Tensor(np.ones((20, 28, 1262), dtype=np.float))


    model_singleLSTM = singleLSTM(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1) 
    model_LSTM = LSTM(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers)
    model_singleGRU = singleGRU(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers) 
    model_GRU = GRU(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers) 
    model_singleRNN = singleRNN(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1) 


    '''
    graph = hl.build_graph(model_singleLSTM, inputs, transforms=transforms)
    graph.save('singleLSTM_hiddenlayer', format='png')

    graph = hl.build_graph(model_LSTM, inputs, transforms=transforms)
    graph.save('LSTM_hiddenlayer', format='png')
    
    graph = hl.build_graph(model_singleGRU, inputs, transforms=transforms)
    graph.save('singleGRU_hiddenlayer', format='png')

    
    graph = hl.build_graph(model_GRU, inputs, transforms=transforms)
    graph.save('GRU_hiddenlayer', format='png')
    
    graph = hl.build_graph(model_singleRNN, inputs, transforms=transforms)
    graph.save('singleRNN_hiddenlayer', format='png')
    
    '''

    for model in [model_singleLSTM, model_LSTM, model_singleGRU, model_GRU, model_singleRNN]:
        summary(model, torch.zeros((20, 28, 1262)))
        
    EDA_input_size = 512
    EDA_num_classes = 512
    encoder_input = torch.Tensor(np.ones((28, 512), dtype=np.float))
    decoder_input = torch.Tensor(np.ones((1, 512), dtype=np.float))

    model = Seq2Seq(28, num_classes=EDA_num_classes, input_size=EDA_input_size, embedding_dim = hidden_size)
    graph = hl.build_graph(model, (encoder_input, decoder_input), transforms=transforms)
    graph.save('EDA_hiddenlayer', format='png')


if __name__ == "__main__":
    main()
