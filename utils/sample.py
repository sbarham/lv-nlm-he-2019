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

# constants
DIVIDER = '------------------------------------------\n'
AVG_LENGTH = None
MIN_LENGTH = None
MAX_LENGTH = None

############################################
#           MAIN SAMPLING CODE             #
############################################

def test_generation(vae, vocab, args, data, epoch):
    # model to eval mode, just in case
    vae.eval()
    
    # initalize the header
    header = None
    if epoch is None:
        header = ['Samples generated from model at {}\n'.format(args.model_path)]
    else:
        header = ['Samples generated at the end of Epoch {}\n'.format(epoch)]
    
    header.append(DIVIDER)
    header.append('\n')
    
    # initialize the results list
    samples = header
    
    # first get some sentences sampled from the prior
    samples += sample_sentences(vae, vocab, args)
    
    # then get a warm interpolation (warm = between corpus sentence embeddings,
    # i.e., z's sampled from two posterior distributions)
    samples += reconstruction(vae, vocab, args, data)
    
    # then get a cold interpolation (cold = between z's sampled from the prior)
    
    # then get sentence reconstructions
    
    # write the results to file
    file_name = None
    if epoch is None:
        file_name = '{}_test_samples'.format(args.dataset)
    else:
        file_name = 'samples_epoch{}'.format(epoch)
    
    file_path = os.path.join(args.save_dir, file_name)
    with open(file_path, 'w') as file:
        file.writelines(samples)

def sample_sentences(vae, vocab, args):
    vae.eval()
    sampled_sents = []
    device = args.device
    
    header = [DIVIDER]
    header.append('\tSamples from the prior\n')
    header.append(DIVIDER)
    
    for i in range(args.num_sentences):
        # sample z from the prior
        z = vae.sample_from_prior(1)
        
        decoded_sentence = z2sent(vae, vocab, z, args)
        
        # append to sampled_sents
        sampled_sents.append(decoded_sentence)
    
    # initialize the results list
    res = header
    
    # join the word-strings for each sentence clean up the results
    for i, sent in enumerate(sampled_sents):
        # prepend the sample number for convenience
        line = '({}): '.format(i) + sent
        
        res.append(line)
        
    res += ['\n\n']
        
    # return the list of sample strings
    return res

def cold_interpolation(vae, vocab, args):
    res = ''
    
    # generate latent code endpoints
    z1 = torch.randn([args.nz]).numpy()
    z2 = torch.randn([args.nz]).numpy()
    
    # create the interpolations
    z = to_var(torch.from_numpy(interpolate(
        start=z1,
        end=z2,
        steps=(args.num_steps - 2)
    ))).float()
    
    # sample sentences from each of the interpolations
    samples, _ = model.inference(z=z)
    
    # create result string
    res += '------- COLD INTERPOLATION --------'
    for line in idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']):
        line = ''.join(line)
        line = clean_sample(line)
        res += '\n' + line
    
    res += '\n\n'
    
    return res

def warm_interpolation(vae, vocab, args, data):
    res = ''
    
    # pick two random sentences
    i = random.randint(0, len(datasets['test']))
    j = random.randint(0, len(datasets['test']))

    # convert the sentences to tensors
    s_i = torch.tensor([datasets['test'][i]['input']], device=device)
    s_i_length = torch.tensor([datasets['test'][i]['length']], device=util.DEVICE)
    s_j = torch.tensor([datasets['test'][j]['input']], device=device)
    s_j_length = torch.tensor([datasets['test'][j]['length']], device=util.DEVICE)

    # encode the two sentences into latent space
    with torch.no_grad():
        _, _, _, z_i = model(s_i, s_i_length)
        _, _, _, z_j = model(s_j, s_j_length)
        z_i, z_j = z_i.cpu(), z_j.cpu()
            
    # create the interpolation
    z1, z2 = z_i.squeeze().numpy(), z_j.squeeze().numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    
    # generate samples from each code point
    samples, _ = model.inference(z=z)
    
    # create the result string
    res += '------- WARM INTERPOLATION --------\n'
    
    res += '(Original 1): '
    line = ''.join(idx2word(
        s_i,
        i2w=i2w,
        pad_idx=w2i['<pad>']
    ))
    line = clean_sample(line)
    res += line + '\n'
    
    for line in idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']):
        line = ''.join(line)
        line = clean_sample(line)
        res += line + '\n'
    
    res += '(Original 2): '
    line = ''.join(idx2word(
        s_j,
        i2w=i2w,
        pad_idx=w2i['<pad>']
    ))
    line = clean_sample(line)
    res += line + '\n\n'
        
    return res

def reconstruction(vae, vocab, args, data):
    # model to eval, just in case
    vae.eval()
    
    # initialize the results list
    res = []
    
    # initialize the sentences data structure we'll use
    sentences = []
    device = args.device
    
    # initialize the header
    header = [DIVIDER]
    header.append('\tReconstructed sentences\n')
    header.append(DIVIDER)
    
    # set results string to the header to start with
    res = header
    
    for _ in range(args.num_reconstructions):    
        # pick short sentences
        s_short = get_random_short(data, args)
        sentences.append({
            'original': s_short,
            'type': 'short'
        })
        
    for _ in range(args.num_reconstructions):
        # pick random sentences
        s_rand = get_random(data, args)
        sentences.append({
            'original': s_rand,
            'type': 'random'
        })
    
    # pick args.num_reconstructions sentences of each type (short, long, random)
    for _ in range(args.num_reconstructions):    
        # pick long sentences
        s_long = get_random_long(data, args)
        sentences.append({
            'original': s_long,
            'type': 'long'
        })
        
    # reconstruct each sentence
    for i, sentence in enumerate(sentences):
        # get the mean and logvar of this each encoded sentence
        with torch.no_grad():
            sent = torch.tensor(sentence['original'], device=args.device)
            sent = sent.unsqueeze(0)
            mean, log_v = vae.encoder(sent)
        
        mean, log_v = mean.cpu()[0], log_v.cpu()[0]
        
        stdev = torch.exp(0.5 * log_v)
        
        # decode the mean sentence
        mean_sentence = z2sent(vae, vocab, mean, args)
        
        # save the mean sentence
        sentence['mean'] = mean
        sentence['mean_sentence'] = mean_sentence
        
        # decode a number of sentences sampled from the latent code
        # (1) sample a latent code
        sentence['random_samples'] = []
        for j in range(3):
            z = torch.randn(args.nz)
            z = z * stdev + mean

            # (2) decode the latent code
            random_sentence = z2sent(vae, vocab, z, args)

            # save the random sentences with their z values
            sentence['random_samples'].append((
                random_sentence,
                z.cpu()
            ))
    
    # create the result string
    for i, sentence in enumerate(sentences):
        ###############################
        # PRINT THE ORIGINAL SENTENCE #
        ###############################
        res.append('\nSentence [{}] ({})\n'.format(i, sentence['type']))
        res.append('(Original):\n\t')
        res.append(idx2sent(sentence['original'], vocab))
        
        ###################################
        # PRINT THE MEAN DECODED SENTENCE #
        ###################################
        res.append('(Mean):\n\t')
        res.append(sentence['mean_sentence'])
        
        ########################################
        # PRINT THE RANDOMLY SAMPLED SENTENCES #
        ########################################
        res.append('(Random Samples):\n')
        for sent, z in sentence['random_samples']:
            # get the distance from the mean and print it in parens
            mean = sentence['mean']
            res.append('({:.3f} from mean)\n\t'.format(np.linalg.norm(mean - z)))
            res.append(sent)
            
        res.append('\n')
        
    res.append('\n\n')
        
    return res



############################################
#            HELPER METHODS                #
############################################

def interpolate(start, end, steps):
    steps = steps + 2
    
    interpolation = np.zeros((start.shape[0], steps))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps)

    return interpolation.T

def idx2sent(idx, vocab):
    # turn a list of idx into a list of string
    sent = vocab.decode_sentence(idx)
    
    # join and clean up the word-strings
    line = ' '.join(sent)
    line = clean_sample(line) + '\n'
    
    # return the clean sentence
    return line

def z2sent(vae, vocab, z, args):
    device = args.device
    
    # shape z
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
    
    # join the words together with space, clean the line, and append '\n'
    line = ' '.join(decoded_sentence)
    line = clean_sample(line) + '\n'
    
    return line

def get_random(data, args):
    s_rand_idx = random.randint(0, len(data) - 1)
    
    return torch.tensor(data[s_rand_idx], device=args.device)

def calc_length_stats(data, args):
    global AVG_LENGTH
    global MAX_LENGTH
    global MIN_LENGTH
    
    # warm up -- get a stochastic average of length, and a stochastic max
    running_length = 0
    running_min = 0
    running_max = 0
    for i in range(args.sample_warmup_period):
        rand_idx = random.randint(0, len(data) - 1)
        rand_sent = data[rand_idx]
        rand_length = len(rand_sent)
        # adjust running sum
        running_length += rand_length
        # adjust running min
        if rand_length < running_min:
            running_min = rand_length
        # adjust running max
        if rand_length > running_max:
            running_max = rand_length
    
    # we want only sentences whose length is roughly in the first sextile
    AVG_LENGTH = running_length / args.sample_warmup_period
    MAX_LENGTH = (AVG_LENGTH + running_min) / 3
    MIN_LENGTH = (AVG_LENGTH + running_max) / 2
    
    print("Stochastic average of dataset length:\t{}".format(AVG_LENGTH))
    print("\tworking max for short sequences:\t{}".format(MAX_LENGTH))
    print("\tworking min for long sequences:\t{}".format(MIN_LENGTH))

def get_random_short(data, args):
    if AVG_LENGTH is None:
        calc_length_stats(data, args)
    
    # find an appropriate sentence
    while True:
        s_rand_idx = random.randint(0, len(data) - 1)
        s_rand_length = len(data[s_rand_idx])
        
        if s_rand_length < MAX_LENGTH:
            s_short = torch.tensor(data[s_rand_idx], device=args.device)
            return s_short
        
def get_random_long(data, args):
    if AVG_LENGTH is None:
        calc_length_stats(data, args)
    
    # find an appropriate sentence
    while True:
        s_rand_idx = random.randint(0, len(data) - 1)
        s_rand_length = len(data[s_rand_idx])
        
        if s_rand_length > MIN_LENGTH:
            s_long = torch.tensor(data[s_rand_idx], device=args.device)
            return s_long
        
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