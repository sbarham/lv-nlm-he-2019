# python
import sys
import os
import time
import importlib
import argparse
import random
import re

# numpy
import numpy as np

# torch
import torch
from torch import nn, optim

# ours
from data import MonoTextData
from modules import VAE
from modules import LSTMEncoder, LSTMDecoder

############################################
#           MAIN SAMPLING CODE             #
############################################

def test_generation(vae, vocab, args, epoch):
    vae.eval()
    samples = []
    
    # first get some sentences sampled from the prior
    samples += sample_sentences(vae, vocab, args)
    
    # then get a warm interpolation (warm = between corpus sentence embeddings,
    # i.e., z's sampled from two posterior distributions)
    
    # then get a cold interpolation (cold = between z's sampled from the prior)
    
    # then get sentence reconstructions
    
    # write the results to file
    file_name = 'samples_epoch{}'.format(epoch)
    file_path = os.path.join(args.save_dir, file_name)
    with open(file_path, 'w') as file:
        file.writelines(samples)

def sample_sentences(vae, vocab, args):
    vae.eval()
    sampled_sents = []
    device = args.device
    
    for i in range(args.num_sentences):
        # sample z from the prior
        z = vae.sample_from_prior(1)
        z = z.view(1,1,-1)
        
        # get the start symbol and end symbol
        start = vocab.word2id['<s>']
        START = torch.tensor([[start]])
        end = vocab.word2id['</s>']
        
        # send both to the proper device
        START = START.to(device)
        z = z.to(device)
        
        # decode z into a sentence
        sentence = vae.decoder.sample_text(START, z, end, device)
        
        # perform idx2word ("decode_sentence") on the sentences and ...
        decoded_sentence = vocab.decode_sentence(sentence)
        
        # append to sampled_sents
        sampled_sents.append(decoded_sentence)
    
    # join the word-strings for each sentence clean up the results
    res = []
    for i, sent in enumerate(sampled_sents):
        line = str(i)
        line += ' '.join(line)
        line = clean_sample(line) + '\n'
        res.append(line)
        
    # return the list of sample strings
    return res



############################################
#            HELPER METHODS                #
############################################

def get_random(datasets, args):
    device = args.device
    
    s_rand_idx = random.randint(0, len(datasets['test']) - 1)    
    s_rand = torch.tensor([datasets['test'][s_rand_idx]['input']], device=device)
    s_rand_len = torch.tensor([datasets['test'][s_rand_idx]['length']], device=device)
    
    return s_rand, s_rand_len

def get_random_short(datasets, args):
    device = args.device
    
    # warm up -- get a stochastic average of length, and a stochastic max
    running_length = 0
    running_min = 0
    for i in range(args.sample_warmup_period):
        s_rand_idx = random.randint(0, len(datasets['test']) - 1)
        s_rand_length = datasets['test'][s_rand_idx]['length']
        running_length += s_rand_length
        if s_rand_length < running_min:
            running_min = s_rand_length
    
    # we want only sentences whose length is roughly in the fourth quartile
    avg_length = running_length / args.sample_warmup_period
    max_length = (avg_length + running_min) / 2
    
    # find an appropriate sentence
    while True:
        s_rand_idx = random.randint(0, len(datasets['test']) - 1)
        s_rand_length = datasets['test'][s_rand_idx]['length']
        
        if s_rand_length < max_length:
            s_short = torch.tensor([datasets['test'][s_rand_idx]['input']], device=device)
            s_short_len = torch.tensor([s_rand_length], device=device)
            return s_short, s_short_len
        
def get_random_long(datasets, args):
    device = args.device
    
    # warm up -- get a stochastic average of length, and a stochastic max
    running_length = 0
    running_max = 0
    for i in range(args.sample_warmup_period):
        s_rand_idx = random.randint(0, len(datasets['test']) - 1)
        s_rand_length = datasets['test'][s_rand_idx]['length']
        running_length += s_rand_length
        if s_rand_length > running_max:
            running_max = s_rand_length
    
    # we want only sentences whose length is roughly in the fourth quartile
    avg_length = running_length / args.sample_warmup_period
    min_length = (avg_length + running_max) / 2
    
    # find an appropriate sentence
    while True:
        s_rand_idx = random.randint(0, len(datasets['test']) - 1)
        s_rand_length = datasets['test'][s_rand_idx]['length']
        
        if s_rand_length > min_length:
            s_long = torch.tensor([datasets['test'][s_rand_idx]['input']], device=device)
            s_long_len = torch.tensor([s_rand_length], device=device)
            return s_long, s_long_len
        
def clean_sample(line):
    # left and right strip the line
    line = line.strip()
    
    # remove leading or trailing reserved symbol
    if line.startswith('<sos>'):
        line = line[5:]
    if line.endswith('<eos>'):
        line = line[:-5]
        
    # again left and right strip the line
    line = line.strip()
    
    # fix the punctuation
    line = clean_punctuation(line)
    
    return line

def clean_punctuation(line):
    # remove space around colons between numbers
    num_colon = re.compile(r'(\d+)\s+:\s+(\d+)')
    line = num_colon.sub(r"\1:\2", line)
    
    # remove space before commas, colons, and periods
    line = re.sub(r"\s+(,|:|\.)\s+", r"\1 ", line)
    
    # remove space around apostrophes
    line = re.sub(r"\s+'\s+", r"'", line)
    
    # remove space before final periods
    line = re.sub(r"\s+\.", r".", line)
    
    return line