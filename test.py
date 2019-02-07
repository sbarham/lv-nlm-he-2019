# python
import argparse
import os
import re
import importlib
import sys

# torch
import torch

# ours
from utils.sample import test_generation
from utils.create import create_corpus, create_model, load_model


############################################
#         INITIALIZE ARGS OBJECT           #
############################################

def init_config():
    parser = argparse.ArgumentParser(description='For testing (i.e., sampling from) pre-trained models')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')

    # sampling constants
    parser.add_argument('--num-sentences', type=int, default=10, help='number of sentences to sample at sample time')
    parser.add_argument('--sample-every', type=int, default=1, help='number of epochs between sample-file generation')
    
    # get the args object
    args = parser.parse_args()
    
    # store True for cuda, False if cuda unavailable
    args.cuda = torch.cuda.is_available()

    # get the directory where the model is stored, make sure it exists
    save_dir = "models/%s" % args.dataset

    if not os.path.exists(save_dir):
        print("No pretrained models available for dataset '{}'".format(args.dataset))
        exit(0)
        
    args.save_dir = save_dir
    model_names = []
    
    files = os.listdir(save_dir)
    for file in files:
        if file.endswith('.pt'):
            model_names.append(file)
    
    model_path = None
    if len(model_names) > 1:
        while model_path is None:
            # print the choices
            print("Choose a model:")
            for i, model_name in enumerate(model_names):
                print("({}) {}".format(i, model_name))
            
            # try to read user input
            choice = input()
            try:
                choice = int(re.findall('\d+', choice)[0])
                model_path = os.path.join(save_dir, model_names[choice])
            except Exception as e:
                print("exception: {}".format(e))
                print("Please enter a valid integer choice from the above.")
    else:
        model_path = os.path.join(save_dir, model_names[0])
        
    args.model_path = model_path
    
    # load the configuration for the given dataset
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)
        
    # print a helpful message and return the args object
    divider()
    print("Found model!")
    print("Using model at '{}'".format(model_path))
    divider()
    
    return args


############################################
#            HELPER METHODS                #
############################################

def divider():
    print("----------------------------------------------")
    
    
############################################
#              MAIN METHOD                 #
############################################
    
def main(args):
    divider()
    print("Using args:")
    print(args)
    divider()
    
    # select device and signal if using cuda or no
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    if args.cuda:
        print('Using cuda')
    
    # prepare dataset, splits, and vocab
    data, vocab = create_corpus(args)
    args.vocab_size = len(vocab)
    train_data, val_data, test_data = data
    
    # create model
    vae = create_model(args, vocab)
    divider()
    print("Model:")
    print(vae)
    divider()
    
    # load the desired state dict
    load_model(vae, args)
    
if __name__ == '__main__':
    args = init_config()
    main(args)