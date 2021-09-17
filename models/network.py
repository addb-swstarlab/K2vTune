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
        return self.x_im

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

        self.emb = nn.Linear(self.input_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, h, eo):
        x = self.emb(x)
        outputs, hidden = self.gru(x, h)
        outputs = self.fc(outputs)
        return outputs, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttnDecoderRNN, self).__init__()
        self.input_dim = input_dim # 1
        self.hidden_dim = hidden_dim # 64
        self.output_dim = output_dim # 1

        self.emb = nn.Linear(self.input_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.attn = nn.Linear(self.hidden_dim*2, 22)
        self.attn_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim)

    def forward(self, x, h, eo):
        # x = (32, 1, 64) = (batch, trg_len, hidden)
        # h = (1, 32, 64) = (layer, batch, hidden)
        x = self.emb(x)
        attn_weights = F.softmax(self.attn(torch.cat((x[:,0], h[0]),1)), dim=1) # (32, 64) cat (32, 64) = (32, 128) => (32, 22)
        # print(attn_weights.shape, eo.shape) # (32, 22) (32, 22, 64) 
        # attn_weights.unsqueeze(1) MAKES (32, 22) => (32, 1, 22)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), eo) # (32, 1, 22) * (32, 22, 64) = (32, 1, 64)
        
        outputs = torch.cat((x[:, 0], attn_applied[:, 0]), 1) # (32, 64) (32, 64) => (32, 128)
        outputs = self.attn_combine(outputs).unsqueeze(0) # (32, 128) => (32, 64) => (1, 32, 64)

        outputs = F.relu(outputs)
        outputs = outputs.permute(1, 0, 2) # (1, 32, 64) -> (32, 1, 64) (batch, len, hidden)
        outputs, hidden = self.gru(outputs, h)

        outputs = self.fc(outputs)
        return outputs, hidden

class GRUNet(nn.Module):
    def __init__(self, encoder, decoder, tf, batch_size):
        super(GRUNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tf = tf # teacher forcing
        self.batch_size = batch_size
        self.trg_len = 4

    def forward(self, x, trg=None):
        if trg is not None:
            self.trg_len = trg.shape[-1] # 4
        self.encoder_outputs, self.encoder_hidden  = self.encoder(x)

        self.decoder_hidden = self.encoder_hidden
        self.decoder_input = torch.zeros((self.batch_size, 1, 1)).cuda()
        self.outputs = torch.zeros(self.batch_size, self.trg_len).cuda()

        for di in range(self.trg_len):
            self.decoder_output, self.decoder_hidden = self.decoder(self.decoder_input, self.decoder_hidden, self.encoder_outputs)
            self.outputs[:,di] = self.decoder_output.squeeze()
            if self.tf:
                self.decoder_input = trg[:,di].view(-1, 1, 1)
            else:
                self.decoder_input = self.decoder_output

        return self.outputs