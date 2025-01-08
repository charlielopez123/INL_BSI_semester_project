import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import utils


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.args = args
        self._input_dim = args.input_dim
        self._num_nodes = args.num_nodes
        self._num_rnn_layers = args.num_rnn_layers
        self._rnn_units = args.hidden_dim
        self._num_classes = args.n_classes
        self._device = torch.device('{}'.format(args.gpu))
                
        self.lstm = nn.LSTM(self._input_dim * self._num_nodes, 
                          self._rnn_units, 
                          self._num_rnn_layers,
                          batch_first=True)
        
        # if args.linear_probing:
        #     for p in self.parameters():
        #         p.requires_grad = False
        

        self.dropout = nn.Dropout(p=args.dropout) # dropout layer before final FC
        self.fc = nn.Linear(self._rnn_units, self._num_classes) # final FC layer
        self.relu = nn.ReLU()

        # print("trainable parameters:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name)  
    
    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, max_seq_len, num_nodes, input_dim)
            seq_lengths: (batch_size, )
        """
        # inputs = inputs.unsqueeze(0)
        inputs = inputs.permute(0,1,3,2)
        batch_size, max_seq_len, _, _ = inputs.shape
        # print("input shape: ",inputs.shape)
        inputs = torch.reshape(inputs, (batch_size, max_seq_len, -1))  # (batch_size, max_seq_len, num_nodes*input_dim)
        
        # initialize hidden states
        initial_hidden_state, initial_cell_state = self.init_hidden(batch_size)

        # LSTM
        if self.args.linear_probing:
            with torch.no_grad():
                output, _ = self.lstm(inputs, (initial_hidden_state, initial_cell_state)) # (batch_size, max_seq_len, rnn_units)
        else:
            output, _ = self.lstm(inputs, (initial_hidden_state, initial_cell_state)) # (batch_size, max_seq_len, rnn_units)
        output = output[:,-1,:]

        # output = output.to(self._device)
        
        # Dropout -> ReLU -> FC
        embedding = self.relu(self.dropout(output))
        logits = self.fc(embedding) # (batch_size, num_classes)
        # logits = logits.squeeze()
        # if(len(logits.shape)<2):
        #     # print("output dim: ",logits.shape)
        #     logits = logits.unsqueeze(0)
        # print("output dim: ",logits.shape)
        if self.task != "SSLEval":
            embedding = embedding.cpu().detach().numpy()
                       
        return logits,embedding    

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        cell = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        return hidden, cell