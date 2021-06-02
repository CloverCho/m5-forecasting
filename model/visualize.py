import numpy as np
import pandas as pd
import torch
import torch.onnx

from torchviz import make_dot
from torchsummary import summary
import hiddenlayer as hl


from model import *



def main():
    transforms = [hl.transforms.Prune('Constant')]

    num_layers = 4
    hidden_size = 512
    
    lstm_input_size = 1262
    lstm_num_classes = 1262
    #lstm_input = pd.DataFrame(np.ones((1262, 28, 3374), dtype=np.float))
    lstm_input = torch.Tensor(np.ones((20, 28, 1262), dtype=np.float))

    model = LSTM(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers)
    
    graph = hl.build_graph(model, lstm_input, transforms=transforms)
    graph.save('lstm_hiddenlayer', format='png')

    
    model = singleLSTM(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1) 
    graph = hl.build_graph(model, lstm_input, transforms=transforms)
    graph.save('singleLSTM_hiddenlayer', format='png')


    model = singleGRU(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers) 
    graph = hl.build_graph(model, lstm_input, transforms=transforms)
    graph.save('singleGRU_hiddenlayer', format='png')


    model = GRU(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers) 
    graph = hl.build_graph(model, lstm_input, transforms=transforms)
    graph.save('GRU_hiddenlayer', format='png')

    model = singleRNN(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=1) 
    graph = hl.build_graph(model, lstm_input, transforms=transforms)
    graph.save('singleRNN_hiddenlayer', format='png')
    
    

if __name__ == "__main__":
    main()