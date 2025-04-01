"""
Implementation of the LFADS encoder-decoder model without fancy priors.
However, the bidirectional GRU is replaced with a Leaky RNN for fun.
"""
import torch
import torch.nn as nn
import torch.functional as F
from omegaconf import OmegaConf

class LeakyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, noise, dt, tau_min, tau_max):
        super().__init__()
        self.hidden_size = hidden_size
        self.noise = noise
        self.in_fc = nn.Linear(input_size, hidden_size)
        self.W = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        nn.init.xavier_normal_(self.W)

        self.tau_range = tau_max - tau_min
        self.tau_min, self.tau_max = tau_min, tau_max
        self.dt = dt
        self.taus = torch.sigmoid(torch.rand(hidden_size)) * self.tau_range + self.tau_min
        self.taus = nn.Parameter(self.taus)
        
        self.tanh = nn.Tanh()

    def recurrence(self, inp_t, h_t):
        tau = torch.sigmoid(self.taus) * self.tau_range + self.tau_min
        alpha = self.dt / tau
        h_t = (1-alpha) * h_t + alpha * self.tanh(self.in_fc(inp_t) + h_t @ self.W + torch.randn_like(h_t) * self.noise)
        return h_t
    
    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, 
                            device=x.device, dtype=x.dtype)
        out = []
        for t in range(x.size(1)):
            h = self.recurrence(x[:, t], h)
            out.append(h)
        h = torch.stack(out, dim=1)
        return h, None
    
class LFADS(nn.Module):
    
    @staticmethod
    def get_default_config(): # model configuration
        conf = OmegaConf.create()
        conf.input_size = None
        conf.generator_size = 128
        conf.controller_size = 16
        conf.latent_size = 32
        conf.dropout = 0.25
        conf.noise_std = 0.01
        conf.dt = 0.02
        conf.tau_min = 0.02
        conf.tau_max = 0.2
        
        return conf
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = LeakyRNN(cfg.input_size, cfg.generator_size,
                                cfg.noise_std, cfg.dt, cfg.tau_min, cfg.tau_max)
        self.generator = LeakyRNN(cfg.generator_size + cfg.controller_size, cfg.generator_size,
                                  cfg.noise_std, cfg.dt, cfg.tau_min, cfg.tau_max)

        self.fc_factors = nn.Sequential(nn.Linear(cfg.generator_size, cfg.latent_size),
                                        nn.Dropout(cfg.dropout))
        self.fc_rates = nn.Linear(cfg.latent_size, cfg.input_size)
        
        self.mu = nn.Linear(cfg.generator_size, cfg.controller_size)
        self.logvar = nn.Linear(cfg.generator_size, cfg.controller_size)
        

        self.inp_dropout = nn.Dropout(cfg.dropout)
        self.enc_dropout = nn.Dropout(cfg.dropout)
        self.gen_dropout = nn.Dropout(cfg.dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        else:
            return mu

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    def forward(self, x):
        x = self.inp_dropout(x)
        h_enc = self.encoder(x)[0]
        h_enc = self.enc_dropout(h_enc)
        
        mu = self.mu(h_enc)
        logvar = self.logvar(h_enc)
        z = self.reparameterize(mu, logvar)
        
        gen_inp = self.gen_dropout(torch.cat([z, h_enc], dim=-1))
        h_gen = self.generator(gen_inp, h_enc[:, -1])[0]
        factors = self.fc_factors(h_gen)
        rates = torch.exp(self.fc_rates(factors))

        kl_loss = self.kl_loss(mu, logvar)
        
        return rates, factors, kl_loss