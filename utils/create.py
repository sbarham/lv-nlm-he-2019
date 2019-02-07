# python
import sys

# torch
import torch
from torch import nn

# ours
from data import MonoTextData
from modules import VAE
from modules import LSTMEncoder, LSTMDecoder


############################################
#              INITIALIZERS               #
############################################

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)

        
############################################
#            CREATE METHODS                #
############################################
        
def create_corpus(args):
    train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)

    print('Train data: %d samples' % len(train_data))
    print('finish reading datasets, vocab size is %d' % len(vocab))
    print('dropped sentences: %d' % train_data.dropped)
    sys.stdout.flush()
    
    return (train_data, val_data, test_data), vocab

def create_model(args, vocab):
    # build initializers
    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    # build encoder
    if args.enc_type == 'lstm':
        encoder = LSTMEncoder(args, args.vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")
    
    # build decoder
    decoder = LSTMDecoder(args, vocab, model_init, emb_init)

    vae = VAE(encoder, decoder, args).to(args.device)
    
    return vae

def load_model(vae, args):
    print("Loading state from indicated model path ...")
    state = None
    if args.cuda:
        state = torch.load(args.model_path)
    else:
        state = torch.load(
            args.model_path,
            map_location=lambda storage, loc: storage
        )
    vae.load_state_dict(state)
    print("...success!")