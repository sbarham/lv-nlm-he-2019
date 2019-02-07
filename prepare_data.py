# python
import requests
import tarfile
import os
import random

# nltk
import nltk
import nltk.corpus as corpus
nltk.download("brown")

def create_brown_data():
    train_file = open('datasets/brown_data/brown.train.txt', 'w')
    test_file = open('datasets/brown_data/brown.test.txt', 'w')
    valid_file = open('datasets/brown_data/brown.valid.txt', 'w')

    train_perc = 0.8

    for sent in corpus.brown.sents():    
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
