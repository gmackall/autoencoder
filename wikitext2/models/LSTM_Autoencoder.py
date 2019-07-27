import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMAutoEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm_down = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstm_up = nn.LSTM(hidden_dim, embedding_dim, bidirectional=True)

        #final loss fn should be
        #(mse/f(input,output)) + sparsity penalty/regularization + evalution of hidden text

        #f: generate probability distribution and minimize catagorical cross entropy between
        #   generated and one hot representation of base

        #sparsity: squared magnitude of code

        #eval

        def loss_fn(inputs, hiddens, outputs):
            out_loss = nn.CrossEntropyLoss()
            middle_loss = nn.CrossEntropyLoss()
            #sparsity here
            return out_loss(inputs, outputs) # + middle_loss
        
        self.loss_fn = loss_fn

    def forward(self, x):
        x = self.lstm_down(x)
        x = self.lstm_up(x)
        return x

    def encode(self, x):
        return self.lstm_down(x)

    def decode(self, x):
        return self.lstm_up(x)
