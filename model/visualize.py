import numpy as np
import pandas as pd
from torchviz import make_dot

from model import *



def main():
    #transforms = [hl.transforms.Prune('Constant')]

    num_layers = 4
    hidden_size = 512
    
    lstm_input_size = 1262
    lstm_num_classes = 1262
    lstm_input = pd.DataFrame(np.ones((1262, 28, 3374), dtype=np.float))

    model = LSTM(num_classes=lstm_num_classes, input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers)
    '''
    graph = hl.build_graph(model, lstm_input, transforms=transforms)

    graph.save('lstm_hiddenlayer', format='png')

    '''
    output = model(lstm_input)

    make_dot(output, params=dict(list(model.named_parameters()))).render("lstm_torchviz", format="png")


if __name__ == "__main__":
    main()