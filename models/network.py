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
        return self.x_im, None

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
        return outputs, hidden, None

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(AttnDecoderRNN, self).__init__()
#         self.input_dim = input_dim # 1
#         self.hidden_dim = hidden_dim # 64
#         self.output_dim = output_dim # 1

#         self.emb = nn.Linear(self.input_dim, self.hidden_dim) # TODO: REMOVE
#         self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
#         self.fc = nn.Linear(self.hidden_dim, self.output_dim)

#         self.attn = nn.Linear(self.hidden_dim*2, 22)
#         self.attn_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim)

#     def forward(self, x, h, eo):
#         # x = (32, 1, 64) = (batch, trg_len, hidden)
#         # h = (1, 32, 64) = (layer, batch, hidden)
#         x = self.emb(x)
#         attn_weights = F.softmax(self.attn(torch.cat((x[:,0], h[0]),1)), dim=1) # (32, 64) cat (32, 64) = (32, 128) => (32, 22)
#         # print(attn_weights.shape, eo.shape) # (32, 22) (32, 22, 64) 
#         # attn_weights.unsqueeze(1) MAKES (32, 22) => (32, 1, 22)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(1), eo) # (32, 1, 22) * (32, 22, 64) = (32, 1, 64)
        
#         outputs = torch.cat((x[:, 0], attn_applied[:, 0]), 1) # (32, 64) (32, 64) => (32, 128)
#         outputs = self.attn_combine(outputs).unsqueeze(0) # (32, 128) => (32, 64) => (1, 32, 64)

#         outputs = F.relu(outputs)
#         outputs = outputs.permute(1, 0, 2) # (1, 32, 64) -> (32, 1, 64) (batch, len, hidden)
#         outputs, hidden = self.gru(outputs, h)

#         # TODO: Attention

#         outputs = self.fc(outputs)
#         return outputs, hidden, attn_weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, attn):
        super(AttnDecoderRNN, self).__init__()
        self.input_dim = input_dim # 1
        self.hidden_dim = hidden_dim # 64
        self.output_dim = output_dim # 1        

        self.emb = nn.Linear(self.input_dim, self.hidden_dim) # TODO: REMOVE
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.attention = attn

    def forward(self, x, h, eo):
        # # x = (32, 1, 64) = (batch, trg_len, hidden)
        # # h = (1, 32, 64) = (layer, batch, hidden)
        # x = self.emb(x)
        # attn_weights = F.softmax(self.attn(torch.cat((x[:,0], h[0]),1)), dim=1) # (32, 64) cat (32, 64) = (32, 128) => (32, 22)
        # # print(attn_weights.shape, eo.shape) # (32, 22) (32, 22, 64) 
        # # attn_weights.unsqueeze(1) MAKES (32, 22) => (32, 1, 22)
        # attn_applied = torch.bmm(attn_weights.unsqueeze(1), eo) # (32, 1, 22) * (32, 22, 64) = (32, 1, 64)
        
        # outputs = torch.cat((x[:, 0], attn_applied[:, 0]), 1) # (32, 64) (32, 64) => (32, 128)
        # outputs = self.attn_combine(outputs).unsqueeze(0) # (32, 128) => (32, 64) => (1, 32, 64)

        # outputs = F.relu(outputs)
        # outputs = outputs.permute(1, 0, 2) # (1, 32, 64) -> (32, 1, 64) (batch, len, hidden)
        # outputs, hidden = self.gru(outputs, h)
        # outputs = self.fc(outputs)

        ### new code ###
        # eo = (32, 22, 128)
        embedded = self.emb(x) # (32, 1, 128)
        if len(embedded.shape) == 2:
            embedded = embedded.unsqueeze(1)
        gru_outputs, hidden = self.gru(embedded, h)
        gru_outputs = gru_outputs.squeeze(1) # (32, 128)

        attn_weights = self.attention(gru_outputs, eo, 1)

        contexts = attn_weights.unsqueeze(1).bmm(eo).squeeze(1) # (32, 1, ??) * (32, 22, 128)

        outputs = self.fc(contexts)

        return outputs, hidden, attn_weights

class Attention(nn.Module):
    """
    https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, batch_size, hidden_size, method="dot", mlp=False):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot': # Luong
            print('attnetion is dot')
            pass
        elif method == 'general': # Luong
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            print('attnetion is general')
        elif method == "concat": # Luong
            self.Wa = nn.Linear(hidden_size*2, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
            print('attnetion is concat')
        elif method == 'bahdanau': # bahdanau
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
            print('attnetion is bahdanau')
        else:
            raise NotImplementedError

        # self.mlp = mlp
        # if mlp:
        #     self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
        #     self.psi = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        batch_size, seq_lens, _ = encoder_outputs.size()
        # if self.mlp:
        #     last_hidden = self.phi(last_hidden)
        #     encoder_outputs = self.psi(encoder_outputs)

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        # if seq_len is not None:
        #     attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim) => (32, 128)
        :param encoder_outputs: (batch_size, max_time, hidden_dim) => (32, 22, 128)
        :param method: str (`dot`, `general`, `concat`, `bahdanau`)
        :return: a score (batch_size, max_time)
        """

        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1) # (32, 128, 1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1) # (32, 22, 128) * (32, 128, 1) => (32, 22, 1) => (32, 22)

        elif method == 'general':
            x = self.Wa(last_hidden) # (32, 128) => (32, 128)
            x = x.unsqueeze(-1) # (32, 128, 1)
            return encoder_outputs.bmm(x).squeeze(-1) # (32, 22, 128)*(32, 128, 1) => (32, 22, 1) => (32, 22)

        elif method == "concat":
            x = last_hidden.unsqueeze(1) # (32, 1, 128)
            x = x.repeat(1, 22, 1) # (32, 22, 128)            
            x = torch.tanh(self.Wa(torch.cat((x, encoder_outputs), 2))) # (32, 22, 128) (32, 22, 128) => (32, 22, 256) => (32, 22, "128")
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1) # (32, 22, 128) * (32, 128, 1) => (32, 22 ,1) => (32, 22)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1) # (32, 1, 128)
            x = x.repeat(1, 22, 1) # (32, 22, 128)
            out = torch.relu(self.Wa(x) + self.Ua(encoder_outputs)) # (32, 22, "128") + (32, 22, "128") => (32, 22, 128)
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1) # (32, 22, 128) * (32, 128, 1) => (32, 22, 1) => (32, 22)

        else:
            raise NotImplementedError

def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len)
    max_idx = max(seq_len)
    for n, idx in enumerate(seq_len):
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs

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
        self.attn_weights = torch.Tensor().cuda()

        for di in range(self.trg_len):
            self.decoder_output, self.decoder_hidden, self.attn_weight = self.decoder(self.decoder_input, self.decoder_hidden, self.encoder_outputs)
            self.outputs[:,di] = self.decoder_output.squeeze()
            if self.tf:
                self.decoder_input = trg[:,di].view(-1, 1, 1)
            else:
                self.decoder_input = self.decoder_output
            if self.attn_weight is not None: # if using attention
                self.attn_weights = torch.cat((self.attn_weights, self.attn_weight.unsqueeze(-1)), dim=2)

        return self.outputs, self.attn_weights