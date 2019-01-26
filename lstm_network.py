import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.autograd import Variable

class SeqModel(nn.Module):
    def __init__(self, args, device):
        super(SeqModel, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.use_gpu = args.cuda
        self.gpunum = args.gpunum
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.embedding_dim = args.embedding_dim
        self.device = device
        self.isbidirectional = args.bidirectional
        self.num_directions = 2 if self.isbidirectional else 1
        self.word_emb = nn.Embedding(args.category_num, args.embedding_dim)
        self.hidden = self.init_hidden()
        self.LSTM_cell = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout, bidirectional=self.isbidirectional)
        self.hidden2tag = nn.Linear(self.hidden_dim, args.category_num)

    def init_hidden(self):
        return (torch.randn(self.num_layers*self.num_directions, self.batch_size/self.gpunum, self.hidden_dim,requires_grad=False).to(self.device),
            torch.randn(self.num_layers*self.num_directions, self.batch_size/self.gpunum, self.hidden_dim,requires_grad=False).to(self.device))
    
    def forward(self, batch_eventid):
        #print("batch_eventid shape:", batch_eventid.shape)
        embeds = self.word_emb(batch_eventid).to(self.device) #[batch_size, seq_len, embedding_dim]
        #print("embeds shape:", embeds.shape)
        lstm_out, self.hidden = self.LSTM_cell(embeds, self.hidden) # lstm_out=>[batch_size, seq_len, hidden_dim]
        #print("lstm_out shape:", lstm_out)
        lstm_out_flatten = lstm_out.contiguous().view(-1, self.hidden_dim)
        #print("lstm_out_flatten shape:", lstm_out_flatten.shape)
        tag_space = self.hidden2tag(lstm_out_flatten) #[batch_size*seq_len, category_num]
        #print("tag_space shape:", tag_space.shape)
        tag_scores = F.log_softmax(tag_space)
        #print("tag_scores shape:", tag_scores.shape)
        return tag_scores
        