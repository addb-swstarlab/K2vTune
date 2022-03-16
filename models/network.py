from this import d
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class RocksDBDataset(Dataset):
    def __init__(self, X, y):
        super(RocksDBDataset, self).__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class SingleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleNet, self).__init__()
        self.input_dim = input_dim # 3327
        self.hidden_dim = hidden_dim # 1024
        self.output_dim = output_dim # 148
        self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
#         self.hidden = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.ReLU())
        self.im_fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x, t=None):
        self.x_kb = self.knob_fc(x)
#         self.h = self.hidden(self.x_kb)
        self.x_im = self.im_fc(self.x_kb)
        return self.x_im, None # matching format

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderRNN, self).__init__()
        self.input_dim = input_dim # 1024
        self.hidden_dim = hidden_dim # 64
        self.linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
    
    def forward(self, x): # x = (batch, seq_len, input_dim)
        x = self.linear(x)
        outputs, hidden = self.gru(x)
        return outputs, hidden

class DecoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DecoderRNN, self).__init__()
        self.input_dim = input_dim # 1
        self.hidden_dim = hidden_dim # 64
        self.output_dim = output_dim # 1

        # self.emb = nn.Linear(self.input_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        # self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, h):
         # x = self.emb(x)
        outputs, hidden = self.gru(x, h)
        # outputs = self.fc(outputs)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, enc_output, dec_output):
        # enc_ouptut : (batch, length, hidden_size) | dec_output : (batch, 1, hidden_size)
        query = self.linear(dec_output.squeeze(1)).unsqueeze(-1) # (batch, hidden_size, 1)
        
        weight = torch.bmm(enc_output, query).squeeze(-1) # (batch, length)
        
        self.weight = self.softmax(weight)
        
        context_vector = torch.bmm(weight.unsqueeze(1), enc_output) # (batch, 1, hidden_size)
        
        return context_vector

class GRUNet(nn.Module):
    def __init__(self, encoder, decoder, tf, batch_size, attention=None, hidden_size=None):
        super(GRUNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tf = tf # teacher forcing
        self.batch_size = batch_size
        self.trg_len = 4
        self.emb_trg = nn.Linear(1, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.attention = attention
        
        if self.attention is not None:    
            self.concat = nn.Linear(hidden_size*2, hidden_size)
            self.relu = nn.ReLU()
            
    def forward(self, x, trg=None):
        if trg is not None:
            self.trg_len = trg.shape[-1] # 4
        self.encoder_outputs, self.encoder_hidden  = self.encoder(x)
        self.decoder_hidden = self.encoder_hidden
        self.bos = torch.zeros((self.batch_size, 1, 1)).cuda()
        self.outputs = torch.zeros(self.batch_size, self.trg_len).cuda()
        if self.tf:
            self.bos_trg = torch.cat((self.bos, trg.unsqueeze(-1)), dim=1) # (batch, trg_len + 1, 1)
            self.embed_trg = self.emb_trg(self.bos_trg) # (batch, trg_len+1 or len, hidden_size)
        else: # for inference
            # self.bos = torch.zeros((self.batch_size, self.trg_len + 1, 1)).cuda() ## ADD
            # self.embed_trg = self.emb_trg(self.bos) ## ADD
            self.decoder_input = self.emb_trg(self.bos)
        
        self.attn_weights = torch.Tensor().cuda()

        for di in range(self.trg_len):
            if self.tf:
                self.decoder_input = self.embed_trg[:, di, :].unsqueeze(1) # (batch, 1, hidden_size)
            # self.decoder_input = self.embed_trg[:, di, :].unsqueeze(1) ## ADD
            self.decoder_output, self.decoder_hidden = self.decoder(self.decoder_input, self.decoder_hidden)
            
            if self.attention is not None:
                self.context_vector = self.attention(self.encoder_outputs, self.decoder_output)
                self.decoder_output = self.relu(self.concat(torch.cat((self.decoder_output, self.context_vector), dim=-1)))
                self.attn_weights = torch.cat((self.attn_weights, self.attention.weight.unsqueeze(-1)), dim=-1)
            
            self.decoder_output = self.fc(self.decoder_output) # (batch, 1, 1)
            self.outputs[:,di] = self.decoder_output.squeeze()
            
            if not self.tf:
                self.decoder_input = self.emb_trg(self.decoder_output)
                # self.bos[:, di+1, :] = self.decoder_output.squeeze(-1) ## ADD
                # self.embed_trg = self.emb_trg(self.bos) ## ADD
                
 
        return self.outputs, self.attn_weights
