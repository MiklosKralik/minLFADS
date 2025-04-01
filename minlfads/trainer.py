"""
Coordinated dropout trainer for the LFADS model.
"""
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from omegaconf import OmegaConf

class Trainer:

    @staticmethod
    def get_default_config():
        conf = OmegaConf.create()
        conf.max_epochs = 300
        conf.batch_size = 32
        conf.cd_ratio = 0.3
        conf.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        conf.num_workers = 1
        
        conf.min_kl_weight = 0.1
        conf.max_kl_weight = 0.9
        conf.kl_anneal_epochs = 10

        conf.lr_init = 1e-3
        conf.lr_min = 1e-6
        conf.lr_step = 0.9

        conf.min_delta = 0.001
        conf.patience_epochs = 3

        conf.log_interval_epochs = 10
        
        conf.best_loss = float('inf')
        conf.best_poisson_loss = float('inf')
        conf.best_kl_loss = float('inf')
        return conf

    def __init__(self, cfg, model, train_dataset, val_dataset): # data is in (Samples, Time, Channels) format
        self.cfg = cfg
        self.model = model.to(cfg.device)
        
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr_init)
        self.poisson_loss = torch.nn.PoissonNLLLoss(log_input=False, full=True)

    @torch.no_grad()
    def coordinated_dropout(self, batch):
        mask = torch.bernoulli(torch.ones_like(batch) * self.cfg.cd_ratio).bool()
        batch_cd = batch.clone().detach()
        batch_cd[mask] = 0
        return batch_cd, mask

    def step(self, batch):
        batch = batch.to(self.cfg.device)
        batch_cd, mask = self.coordinated_dropout(batch)

        rates, factors, kl_loss = self.model(batch_cd)
        poisson_loss = self.poisson_loss(rates[mask], batch[mask])
        loss = poisson_loss + kl_loss
        return loss, poisson_loss, kl_loss

    def run(self):
        for epoch in range(self.cfg.max_epochs):
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                loss, poisson_loss, kl_loss = self.step(batch)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                for val_batch in self.val_loader:
                    val_loss, val_poisson_loss, val_kl_loss = self.step(val_batch)
            print('val_loss:', val_loss.item(), 'val_poisson_loss:', val_poisson_loss.item(), 'val_kl_loss:', val_kl_loss.item())













