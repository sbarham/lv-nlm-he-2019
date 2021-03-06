# python
import requests
import tarfile
import os
import random
from collections import Counter

# nltk
import nltk
import nltk.corpus as corpus
nltk.download("brown")

RANDOM_SEED = 911
BROWN_VOCAB_SIZE = 20_000

def create_vocab():
    sents = corpus.brown.sents()
    words = [word for sent in sents for word in sent]
    vocab = Counter(words).most_common(BROWN_VOCAB_SIZE)
    vocab = [word[0] for word in vocab]
    
    return vocab

def create_brown_data():
    random.seed(RANDOM_SEED)
    
    train_file = open('datasets/brown_data/brown.train.txt', 'w')
    test_file = open('datasets/brown_data/brown.test.txt', 'w')
    valid_file = open('datasets/brown_data/brown.valid.txt', 'w')

    vocab = create_vocab()
    
    train_perc = 0.8

    for sent in corpus.brown.sents():
        # replace out-of-vocab words with '_UNK'
        sent = [word if word in vocab else '_UNK' for word in sent]
        
        # join the words using ' ', strip extraneous whitespace, and add a \n
        sent = ' '.join(sent)
        sent = sent.strip()
        sent += '\n'

        # prepend 0\t, since the code seems to require it
        sent = '0\t' + sent

        choice = random.uniform(0, 1)

        if choice < train_perc:
            train_file.write(sent)
        elif choice < (train_perc + 0.1):
            test_file.write(sent)
        else:
            valid_file.write(sent)

    train_file.close()
    test_file.close()
    valid_file.close()

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = "1yH3Ufj20R06OX1prPMPS6E3fKSG_2MqN"
    destination = "datasets.tar.gz"
    download_file_from_google_drive(file_id, destination)

    # theirs
    tar = tarfile.open(destination, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(destination)
    
    # our brown data
    create_brown_data()
