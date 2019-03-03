import argparse
import os
import random

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

from GPT2.model import GPT2LMHeadModel
from GPT2.opt import OpenAIAdam
from GPT2.encoder import get_encoder
from GPT2.utils import load_weight
from GPT2.config import GPT2Config

from sacred import Experiment
from sacred.observers import MongoObserver

from tqdm import tqdm

ex = Experiment('gpt-2-finetune')

ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='experiments'))


@ex.config
def config():
    config = dict(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,

        batch_size=48,
        epochs=1,
        trainctxsize=64,
        quit_after=10000,

        lr=6.25e-7,
        lr_schedule='warmup_linear',
        lr_warmup=6.25e-7,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        l2=0.01,
        vector_l2=False,
        max_grad_norm=1
    )


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, f, enc, maxlen=256):
        with open(f) as fh:
            self.content = enc.encode(fh.read())
        self.maxlen = maxlen
        self.enc = enc

    def __len__(self):
        return len(self.content) - self.maxlen + 1

    def __getitem__(self, index):
        context_tokens = self.content[index:index + self.maxlen]
        context_tokens = torch.tensor(context_tokens, dtype=torch.long)
        next_tok = self.content[index]
        next_tok = torch.tensor(next_tok, dtype=torch.long)

        return context_tokens, next_tok


@ex.main
def main(_run, config):
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load(
            'gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.train()
    encoder = get_encoder()

    dataset = ChunkDataset('wpdumptiny.txt', encoder, config['trainctxsize'])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    loss_fn = nn.NLLLoss()

    n_updates_total = len(loader) * config['epochs']
    print('Will do', n_updates_total, 'updates')
    opt = OpenAIAdam(model.parameters(),
                     lr=config['lr'],
                     schedule=config['lr_schedule'],
                     warmup=config['lr_warmup'],
                     t_total=n_updates_total,
                     b1=config['beta1'],
                     b2=config['beta2'],
                     e=config['epsilon'],
                     l2=config['l2'],
                     vector_l2=config['vector_l2'],
                     max_grad_norm=config['max_grad_norm'])

    step = 0
    for epoch in range(config['epochs']):
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)

            if step == config['quit_after']:
                torch.save(model.state_dict(),
                    'gpt2-writingprompts-pytorch_model.bin')

                break

            opt.zero_grad()
            logits, past = model(x)
            logits = logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            loss = loss_fn(log_probs, y)
            loss.backward()
            opt.step()
            tqdm.write('Loss: %.5f' % loss)
            _run.log_scalar('loss', float(loss.item()), step)
            step += 1
if __name__ == '__main__':
    ex.run_commandline()
