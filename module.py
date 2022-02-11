import torch
from torch import nn
from torch.autograd import grad
from torch.optim import Adam
from torch.nn import functional as F
from optimization import BertAdam

from matplotlib import pyplot as plt


class IntensityNet(nn.Module):

    def __init__(self, config):
        super(IntensityNet, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=1)
        self.linear2 = nn.Linear(in_features=config.hid_dim+1, out_features=config.mlp_dim)
        self.module_list = nn.ModuleList([nn.Linear(in_features=config.mlp_dim, out_features=config.mlp_dim) for _ in range(config.mlp_layer-1)])
        self.linear3 =  nn.Sequential(nn.Linear(in_features=config.mlp_dim, out_features=1), nn.Softplus())

        self.mean_first = config.mean_first
        self.log_t = config.log_t


    def forward(self, hidden_state, target_time):
        eps = 1e-10

        for p in self.parameters():
            p.data *= (p.data>=0)

        target_time.requires_grad_(True)
        if self.log_t:
            target_time = torch.log(target_time+eps)
        t = self.linear1(target_time.unsqueeze(dim=-1))

        out = torch.tanh(self.linear2(torch.cat([hidden_state[:,-1,:], t], dim=-1)))
        for layer in self.module_list:
            out = torch.tanh(layer(out))
        int_lmbda = F.softplus(self.linear3(out))
        int_lmbda_mean = int_lmbda.mean()

        lmbda = grad(
            int_lmbda.mean(), 
            target_time, 
            create_graph=True, retain_graph=True)[0]
        log_lmbda = (lmbda + eps).log()
        log_lmbda_mean = log_lmbda.mean()

        if self.mean_first:
            nll = int_lmbda_mean - log_lmbda_mean
        else:
            nll = (int_lmbda - log_lmbda).mean()        

        return [nll, log_lmbda_mean, int_lmbda_mean, lmbda]

class GTPP(nn.Module):

    def __init__(self, config):

        super(GTPP, self).__init__()

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_mode = config.log_mode # TODO mean to be used here?


        self.embedding = nn.Embedding(num_embeddings=config.event_class, embedding_dim=config.emb_dim)
        self.emb_drop = nn.Dropout(p=config.dropout)
        self.lstm = nn.LSTM(input_size=1+config.emb_dim,
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False)
        self.intensity_net = IntensityNet(config)
        self.set_optimizer(total_step=1)


    def set_optimizer(self, total_step, use_bert=False):
        if use_bert:
            self.set_optimizer = BertAdam(params=self.parameters(),
                                          lr=self.lr,
                                          warmup=0.1,
                                          t_total=total_step)
        else:
            self.set_optimizer = Adam(self.parameters(), lr=self.lr)


    def forward(self, batch):
        time_seq, event_seq = batch
        event_seq = event_seq.long()
        emb = self.embedding(event_seq)
        emb = self.emb_drop(emb)
        lstm_input = torch.cat([emb, time_seq.unsqueeze(-1)], dim=-1)
        hidden_state, _ = self.lstm(lstm_input)

        nll, log_lmbda, int_lmbda, lmbda = self.intensity_net(hidden_state, time_seq[:, -1])

        return [nll, log_lmbda.detach(), int_lmbda.detach(), lmbda.detach()]


    def train_batch(self, batch):

        self.set_optimizer.zero_grad()
        nll, log_lmbda, int_lmbda, lmbda = self.forward(batch)
        loss = nll
        loss.backward()
        self.set_optimizer.step()

        return nll.item(), log_lmbda.item(), int_lmbda.item(), lmbda











