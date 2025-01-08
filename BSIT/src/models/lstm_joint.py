import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import utils
import numpy as np


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
                
        self._input_dim = args.input_dim
        self._output_dim = args.output_dim
        self._num_nodes = args.num_nodes
        self._num_rnn_layers = args.num_rnn_layers
        self._rnn_units = args.hidden_dim
        self._num_classes = args.n_classes
        self._device = torch.device('{}'.format(args.gpu))
        self.momentum = args.cluster_momentum
                
        self.encoder = nn.LSTM(self._input_dim * self._num_nodes, 
                          self._rnn_units, 
                          self._num_rnn_layers,
                          batch_first=True)

        self.decoder_in = nn.LSTM(self._input_dim * self._num_nodes, 
                          self._rnn_units, 
                          1,
                          batch_first=True)
        
        self.decoder_out = nn.LSTM(self._rnn_units, 
                          self._rnn_units, 
                          self._num_rnn_layers-1,
                          batch_first=True)

        self.dropout = nn.Dropout(p=args.dropout) # dropout layer before final FC
        self.fc = nn.Linear(self._rnn_units, self._num_classes) # final FC layer
        self.relu = nn.ReLU()
        self.projection_layer = nn.Linear(self._rnn_units, self._output_dim* self._num_nodes)
        self.dropout2 = nn.Dropout(p=args.dropout)  # dropout before projection layer

        n_embedding = self._num_classes
        if self._num_classes ==1:
            n_embedding = 2

        self.embedding_dict = torch.zeros(n_embedding, self._rnn_units).to(self._device)
        # self.embedding_dict.requires_grad = False


    def update_embedding(self, embeddings, labels):
        for label_id in range(self._num_classes):
            index_i = [i for i, item in enumerate(labels) if item == label_id]
            if len(index_i) == 0:
                continue
            # print("indexes and label: ",label_id, index_i )
            embedding_i = embeddings[index_i]
            average_batch_embedding_i = torch.mean(embedding_i,dim=0)
            # print("average label i: ",average_batch_embedding_i)
            self.embedding_dict[label_id] = (1 - self.momentum) * self.embedding_dict[label_id] + self.momentum * average_batch_embedding_i
            # self.embedding_dict = self.embedding_dict.detach()
   
    
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
        # print("input:")
        # print(inputs)
        # print(initial_hidden_state)
        # print(initial_cell_state)

        # LSTM
        final_hidden, encoder_hidden_state = self.encoder(inputs, (initial_hidden_state, initial_cell_state)) # (batch_size, max_seq_len, rnn_units)
        # print(final_hidden)
        # print(encoder_hidden_state)
        output = final_hidden[:,-1,:]

        decoder_initial = torch.zeros(
            (batch_size,
             1,
             self._num_nodes *
             self._output_dim)).to(
            self._device)
        outputs = torch.zeros(
            max_seq_len,
            batch_size,
            self._num_nodes *
            self._output_dim).to(
            self._device)
        
        # print(initial_hidden_state.shape,initial_cell_state.shape,encoder_hidden_state[0].shape,encoder_hidden_state[1].shape)

        current_input = decoder_initial
        for t in range(max_seq_len):
            # print(current_input.shape)
            output_t,_ = self.decoder_in(current_input, (encoder_hidden_state[0][0].unsqueeze(0),encoder_hidden_state[1][0].unsqueeze(0)))
            current_input = output_t
            output_t,_ = self.decoder_out(current_input, (encoder_hidden_state[0][1:],encoder_hidden_state[1][1:]))
            # print("output: ",output_t.shape)
            projected = self.projection_layer(self.dropout2(
                output_t.reshape(batch_size, -1)))
            projected = projected.reshape((batch_size, self._num_nodes * self._output_dim))
            outputs[t] = projected
            current_input = projected.reshape((batch_size, 1, self._num_nodes * self._output_dim))

        # output = output.to(self._device)
        # print("output: ")
        # print(output)
        # Dropout -> ReLU -> FC
        embedding = self.relu(self.dropout(output))
        # print(embedding)
        # input()
        # print(inputs.shape,outputs.shape)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)
        outputs = outputs.reshape((batch_size, max_seq_len,self._output_dim,self._num_nodes,))
        logits = self.fc(embedding)
                       
        return outputs, logits, embedding

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        cell = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        return hidden, cell