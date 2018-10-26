import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class MNIST_Dataset(Dataset):    
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return np.random.binomial(1, self.image[idx, :]).astype('float32')
    '''
    ramdomly binarized MNIST
    '''

    
class Encoder(nn.Module):
    '''
    encoder
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        input_sim = 784
        hidden_dim = 200
        output_dim = 50
        '''
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transform = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Tanh())
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, output_dim) #log_sd
    
    def forward(self, x):
        out = self.transform(x)
        mu = self.fc_mu(out)
        logsigma = self.fc_logsigma(out)
        sigma = torch.exp(logsigma)
        return mu, sigma
    
class IWAE_1(nn.Module):
    def __init__(self, dim_h1, dim_image_vars,stick = True):
        super(IWAE_1, self).__init__()
        self.stick = stick
        self.dim_h1 = dim_h1
        self.dim_image_vars = dim_image_vars

        ## encoder
        self.encoder_h1 = Encoder(dim_image_vars, 200, dim_h1)
        
        ## decoder
        self.decoder_x =  nn.Sequential(nn.Linear(dim_h1, 200),
                                        nn.Tanh(),
                                        nn.Linear(200, 200),
                                        nn.Tanh(),
                                        nn.Linear(200, dim_image_vars),
                                        nn.Sigmoid())
        
    def encoder(self, x):
        mu_h1, sigma_h1 = self.encoder_h1(x)
        eps = Variable(sigma_h1.data.new(sigma_h1.size()).normal_())
        h1 = mu_h1 + sigma_h1 * eps                
        return h1, mu_h1, sigma_h1
    
    def decoder(self, h1):
        p = self.decoder_x(h1)
        return p
    
    def forward(self, x):
        h1, mu_h1, sigma_h1 = self.encoder(x)
        p = self.decoder(h1)
        return (h1, mu_h1, sigma_h1), (p)

    def train_loss(self, inputs):
        h1, mu_h1, sigma_h1 = self.encoder(inputs)
        if self.stick==True:
            log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1.detach())/sigma_h1.detach())**2 - torch.log(sigma_h1.detach()), -1)
        else:
            log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        '''
        log_posterior
        this has size [k,batch_size]
        '''
        p = self.decoder(h1)
        log_Ph1 = torch.sum(-0.5*h1**2, -1)
        '''
        log_prior
        this has size [k,batch_size]
        '''
        
        #log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)
        log_PxGh1 = -torch.sum(F.binary_cross_entropy(p, inputs, reduction='none'), -1)
        '''
        log likelihod for the decoder, which is a log likelihood over bernoulli
        this has size [k,batch_size]
        '''

        log_weight = log_Ph1 + log_PxGh1 - log_Qh1Gx
        '''
        matrix of log(w_i)
        this has size [k,batch_size]
        '''
        
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        '''
        normalize to prevent overflow
        maximum w's for each batch element, where maximum is taking over k samples from posterior
        
        Note: For plian version of VAE, this is identically 0
        '''
        
        weight = torch.exp(log_weight)
        '''
        exponential the log back to get w_i's
        Note: For plian version of VAE, this is identically 1
        '''
        
        weight = weight / torch.sum(weight, 0)
        '''
        \tilda(w_i)
        Note: For plian version of VAE, this is identically 1
        '''
       
        weight = Variable(weight.data, requires_grad = False)
        '''
        stop gradient on \tilda(w)
        '''
        
        loss = -torch.mean(torch.sum(weight * (log_Ph1 + log_PxGh1 - log_Qh1Gx), 0))
        return loss

    def test_loss(self, inputs):
        h1, mu_h1, sigma_h1 = self.encoder(inputs)
        log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)        
        p = self.decoder(h1)
        log_Ph1 = torch.sum(-0.5*h1**2, -1)
        #log_PxGh1 = torch.sum(inputs*torch.log(p) + (1-inputs)*torch.log(1-p), -1)
        log_PxGh1 = -torch.sum(F.binary_cross_entropy(p, inputs, reduction='none'), -1)
        log_weight = log_Ph1 + log_PxGh1 - log_Qh1Gx
        weight = torch.exp(log_weight)
        loss = -torch.mean(torch.log(torch.mean(weight, 0)))        
        return loss

    