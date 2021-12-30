###################################################
# Nicolo Savioli, 2017 -- Conv-GRU pytorch v 1.0  #
###################################################
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    
    def __init__(self,input_size,hidden_size,kernel_size):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = True
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((self.kernel_size - 1) / 2)
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,2 * self.hidden_size,self.kernel_size,padding=self.padding)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,self.hidden_size,self.kernel_size,padding=self.padding)
        dtype            = torch.FloatTensor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self,input,hidden):
        if hidden is None:

           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           if self.cuda_flag  == True:

              hidden    = Variable(torch.zeros(size_h)).cuda() 
           else:
              hidden    = Variable(torch.zeros(size_h))
        #print('input type:', (input[0,1,1,1]), hidden[0,1,1,1])
        c1           = self.ConvGates(torch.cat((input,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = torch.sigmoid(rt)
        update_gate  = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = torch.tanh(p1)
        #next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        next_h = (1-update_gate) *hidden +  torch.mul(update_gate , ct)
        return next_h
