
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
device = torch.device("cuda:0")
class NSMC_RNN(nn.Module):
    def __init__(self, num_vocab, embedding_dim, hidden_dim, num_layers, out_node, drop_percent=0.2): 
        super().__init__()

        self.num_vocab = num_vocab
        self.embed = nn.Embedding(num_embeddings=num_vocab, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_percent)
        self.fc = nn.Linear(hidden_dim, out_node) # binary이기 때문에 out_node = 1 (sigmoid를 통해 0,1사이 값 출력)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.embed(x)
        out, hidden = self.rnn(x)
        out = self.dropout(out)
        out = self.fc(out[:,-1,:])
        out = self.sigmoid(out)
        return out


class NSMC_LSTM(nn.Module):
    def __init__(self, num_vocab, embedding_dim, hidden_dim, num_layers, out_node, drop_percent=0.2): 
        super().__init__()

        self.num_vocab = num_vocab
        self.embed = nn.Embedding(num_embeddings=num_vocab, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_percent)
        self.fc = nn.Linear(hidden_dim, out_node) # binary이기 때문에 out_node = 1 (sigmoid를 통해 0,1사이 값 출력)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.embed(x)
        out, hidden = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:,-1,:])
        out = self.sigmoid(out)
        return out


class NSMC_GRU(nn.Module):
    def __init__(self, num_vocab, embedding_dim, hidden_dim, num_layers, out_node, drop_percent=0.2): 
        super().__init__()

        self.num_vocab = num_vocab
        self.embed = nn.Embedding(num_embeddings=num_vocab, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_percent)
        self.fc = nn.Linear(hidden_dim, out_node) # binary이기 때문에 out_node = 1 (sigmoid를 통해 0,1사이 값 출력)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.embed(x)
        out, hidden = self.gru(x)
        out = self.dropout(out)
        out = self.fc(out[:,-1,:])
        out = self.sigmoid(out)
        return out

class LSTM_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_layers,  output_dim, embedding_dim,drop_percent=0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_percent)
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        embed = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embed) #hidden size = 1 x batch x hidden size    
        attention_output = self.attention(output, hidden)

        output = self.linear(attention_output)
        output = self.sigmoid(output)
        return output
  
    def attention(self, lstm_output, final_hidden):

#lstm_output : batch x seq len x hidden
#final_hidden : 1 x batch x hidden

        final_hidden = final_hidden[-1:,].squeeze(0)
    # final_hidden = batch x hidden

    # torch.bmm(lstm_output, final_hidden.unsqueeze(2)) -> size : batch x seq len x 1
        attention_output = torch.bmm(lstm_output, final_hidden.unsqueeze(2)).squeeze(2)
    # attention_output = batch x seq len 
    
        attention_score = F.softmax(attention_output,1) # 가로(seq len)에 대해 softmax
    # attention_score = batch x seq len

        final_score = torch.bmm(lstm_output.transpose(1,2), attention_score.unsqueeze(2)).squeeze(2)
    # final_score = batch x hidden

        return final_score
 
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size,
        
        
         self.hidden_dim).zero_()
 

