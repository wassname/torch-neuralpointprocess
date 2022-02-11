# %reload_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12.0, 3)

import numpy as np
import tqdm
import torch

from argparse import ArgumentParser

from torch.utils.data import DataLoader

from utils import read_timeseries,generate_sequence, plt_lmbda
from module import GTPP
from run import get_parser





# +
parser = get_parser()
config = parser.parse_args([])

path = 'data/'

if config.data == 'exponential_hawkes':

    train_data = read_timeseries(path + config.data + '_training.csv')
    val_data = read_timeseries(path + config.data + '_validation.csv')
    test_data = read_timeseries(path + config.data + '_testing.csv')
else:
    raise NotImplemented('only exponential_hawkes')



train_timeseq, train_eventseq = generate_sequence(train_data, config.seq_len, log_mode=config.log_mode)
train_loader = DataLoader(torch.utils.data.TensorDataset(train_timeseq, train_eventseq), shuffle=True, batch_size=config.batch_size)
val_timeseq, val_eventseq = generate_sequence(val_data, config.seq_len, log_mode=config.log_mode)
val_loader = DataLoader(torch.utils.data.TensorDataset(val_timeseq, val_eventseq), shuffle=False, batch_size=len(val_data))

model = GTPP(config)

best_loss = 1e3
patients = 0
tol = 333

for epoch in range(config.epochs):

    model.train()

    loss1 = loss2 = loss3 = 0

    for batch in train_loader:
        loss, log_lmbda, int_lmbda, lmbda = model.train_batch(batch)

        loss1 += loss
        loss2 += log_lmbda
        loss3 += int_lmbda


    model.eval()

    for batch in val_loader:
        val_loss, val_log_lmbda, val_int_lmbda, _ = model(batch)

    if best_loss > val_loss:
        best_loss = val_loss.item()
    else:
        patients += 1
        if patients >= tol:
            print("Early Stop")
            print("epoch", epoch)
            plt_lmbda(train_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)
            break

    if epoch % config.prt_evry == 0:
        print("Epochs:{}".format(epoch))
        print("Training  : Negative Log Likelihood:{:2.6f}   Log Lambda:{:2.6f}:   Integral Lambda:{:2.6f}".format(loss1/train_timeseq.size(0), -loss2 / train_timeseq.size(0), loss3 / train_timeseq.size(0)))
        print("Validation: Negative Log Likelihood:{:2.6f}   Log Lambda:{:2.6f}:   Integral Lambda:{:2.6f}".format(val_loss / val_timeseq.size(0),
                                                                                        -val_log_lmbda / val_timeseq.size(0),
                                                                                        val_int_lmbda/val_timeseq.size(0)))
        plt_lmbda(train_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)
        plt_lmbda(test_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)


print("end")
# -


# +
# class CryptoTraderPL_NLL(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self._model = GTPP(config)

#     def forward(self, x):
#         return self._model(x)

#     def training_step(self, batch, batch_idx, phase='train'):
#         """
#         Training step which runs for N steps, and get loss over all of them
#         """
#         x, l, r = batch
#         y_pred = self._model(x)
        
#         # we have multiple targets. So move them to batch
#         l2 = l.reshape(-1)
#         y_pred2 = y_pred.reshape((*l2.shape, 3))
#         loss = F.nll_loss(y_pred2, l2)

#         # record weights
#         self.log_dict({
#             f'loss/{phase}': loss,
#         }, prog_bar=True)

#         assert torch.isfinite(loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         return self.training_step(batch, batch_idx, phase='val')
    
#     def predict_step(self, batch, batch_idx):
#         x, y, r = batch
#         y_pred = self.forward(x)
#         return y_pred, y, r

#     def configure_optimizers(self):
#         optim = Ranger21(self.parameters(),
#                          lr=self.train_kwargs['lr'],
#                          num_epochs=num_epochs,
#                          num_batches_per_epoch=num_batches_per_epoch,
#                          weight_decay=self.train_kwargs['weight_decay'])
#         return {'optimizer': optim, 'monitor': 'loss/val'}
# -













