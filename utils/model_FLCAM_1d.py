#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        return res





class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class FreqChanGate(nn.Module):
    def __init__(self, gate_channel):
        super(FreqChanGate, self).__init__()
        self.gate_s = nn.Sequential(
            nn.Conv1d(gate_channel, gate_channel//16, 3, stride=1, padding=1),
            nn.BatchNorm1d(gate_channel//16),
            nn.ReLU(),
            nn.Conv1d(gate_channel//16, 1, 3, stride=1, padding=1),
        )      
    def forward(self, in_tensor):
        x = in_tensor.permute(0,1,3,2)
        out_tensor = self.gate_s(torch.mean(in_tensor, dim=2))  
        out_tensor = out_tensor.unsqueeze(2)
        out_tensor = out_tensor.expand_as(in_tensor)
        return out_tensor
    
    
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.FreqChan_att = FreqChanGate(gate_channel)
    def forward(self,in_tensor):
        att  = 1+ F.sigmoid( self.FreqChan_att(in_tensor)) 
        return att * in_tensor



class CNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 atte_layers=[0, 1, 1, 1, 1, 1, 1],
                 pooling_layers = [1, 1, 1, 1, 1, 0, 0],
                 temperature=31,
                 pool_dim='freq'):
        super(CNN, self).__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
            
            cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))
                
                
            if atte_layers[i] == 1:
                cnn.add_module("attention{0}".format(i), BAM(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)

            if pooling_layers[i] == 1:
                cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):    #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class BiGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        #self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x


class CRNN(nn.Module):
    def __init__(self,
                 n_input_ch,
                 n_class=10,
                 activation="glu",
                 conv_dropout=0.5,
                 
                 n_RNN_cell=128,
                 n_RNN_layer=2,
                 rec_dropout=0,
                 attention=True,
                 **convkwargs):
        super(CRNN, self).__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
        self.AvgPooling = nn.AvgPool2d((1,4))
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            if self.attention == "time":
                self.softmax = nn.Softmax(dim=1)          # softmax on time dimension
            elif self.attention == "class":
                self.softmax = nn.Softmax(dim=-1)         # softmax on class dimension

    def forward(self, x): #input size : [bs, freqs, frames]
        #cnn
        if self.n_input_ch > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1) #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        # pooling 
        x = self.AvgPooling(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
#             print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(bs, frame, ch*freq)   # x.contiguous.view(bs, frame, ch*freq) 
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1) # x size : [bs, frames, chan]

        #rnn
        x = self.rnn(x) #x size : [bs, frames, 2 * chan]
        x = self.dropout(x)

        #classifier
        strong = self.dense(x) #strong size : [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x) #sof size : [bs, frames, n_class]
            sof = self.softmax(sof) #sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1) # [bs, n_class]
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak



