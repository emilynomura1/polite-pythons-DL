import numpy as np
import tensorflow as tf
from collections import Counter
#from attenvis import AttentionVis
#av = AttentionVis()
from get_final_data import get_final_data

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
NON_POLITE_WINDOW_SIZE = 52
POLITE_WINDOW_SIZE = 52
##########DO NOT CHANGE#####################

def pad_corpus(non_polite, polite):
    """
	DO NOT CHANGE:
	arguments are lists of non_polite, polite sentences. Returns [non_polite-sents, polite-sents]. The
	text is given an initial "*STOP*". polite is padded with "*START*" at the beginning for Teacher Forcing.
	:param non_polite: list of non_polite sentences
	:param non_polite: list of non_polite sentences
	:param polite: list of polite sentences
	:return: A tuple of: (list of padded sentences for non_polite, list of padded sentences for polite)
	"""
    non_polite_padded_sentences = []
    for line in non_polite:
        padded_non_polite = line[:NON_POLITE_WINDOW_SIZE]
        padded_non_polite += [STOP_TOKEN] + [PAD_TOKEN] * (NON_POLITE_WINDOW_SIZE - len(padded_non_polite)-1)
        non_polite_padded_sentences.append(padded_non_polite)
        polite_padded_sentences = []
    for line in polite:
        padded_polite = line[:POLITE_WINDOW_SIZE]
        padded_polite = [START_TOKEN] + padded_polite + [STOP_TOKEN] + [PAD_TOKEN] * (POLITE_WINDOW_SIZE - len(padded_polite)-1)
        polite_padded_sentences.append(padded_polite)
    return non_polite_padded_sentences, polite_padded_sentences

def build_vocab(sentences):
    """
	DO NOT CHANGE
    Builds vocab from list of sentences
	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
    tokens = []

    for s in sentences: 
        tokens.extend(s)
    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))
    # flat_list = [item for sublist in sentences for item in sublist]
    
    # x = Counter(flat_list)
    # y = dict((k,v) for k,v in x.items() if v>2)
    # keep_words = list(y.keys())
    # all_words_new = []
    # for word in all_words:
    #     if word in keep_words:
    #         all_words_new.append(word)
    vocab =  {word:i for i,word in enumerate(all_words)}
    # vocab["*UNK*"] = len(vocab)
    return vocab, vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
    """
	DO NOT CHANGE
    Convert sentences to indexed
	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    # print(len(sentences), print(len(vocab)))
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

#@av.get_data_func
def get_data():
    """
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.
	:param non_polite_training_file: Path to the non_polite training file.
	:param polite_training_file: Path to the polite training file.
	:param non_polite_test_file: Path to the non_polite test file.
	:param polite_test_file: Path to the polite test file.
	:return: Tuple of train containing:
	(2-d list or array with polite training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with polite test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with non_polite training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with non_polite test sentences in vectorized/id form [num_sentences x 14]),
	polite vocab (Dict containg word->index mapping),
	non_polite vocab (Dict containg word->index mapping),
	polite padding ID (the ID used for *PAD* in the polite vocab. This will be used for masking loss)\
    """

	#1) Read tokenized polite and non_polite Data for training and testing
    non_polite_train, polite_train, non_polite_test, polite_test = get_final_data() #change this later
    
	#2) Pad training data (see pad_corpus)
    non_polite_train_padded, polite_train_padded = pad_corpus(non_polite_train, polite_train)

	#3) Pad testing data (see pad_corpus)
    non_polite_test_padded, polite_test_padded = pad_corpus(non_polite_test, polite_test)

	#4) Build vocab for non_polite (see build_vocab)
    non_polite_vocab, non_polite_pad_token_idx = build_vocab(non_polite_train_padded)

	#5) Build vocab for polite (see build_vocab)
    polite_vocab, polite_pad_token_idx = build_vocab(polite_train_padded)

	#6) Convert training and testing polite sentences to list of IDS (see convert_to_id)
    polite_train_ids = convert_to_id(polite_vocab, polite_train_padded)
    polite_test_ids = convert_to_id(polite_vocab, polite_test_padded)

	#7) Convert training and testing non_polite sentences to list of IDS (see convert_to_id)
    non_polite_train_ids = convert_to_id(non_polite_vocab, non_polite_train_padded)
    non_polite_test_ids = convert_to_id(non_polite_vocab, non_polite_test_padded)
    
    return polite_train_ids, polite_test_ids, non_polite_train_ids, non_polite_test_ids, polite_vocab, non_polite_vocab, polite_pad_token_idx

print("Running preprocessing...")
# data_dir   = '../tag-and-generate-data-prep/data/catcher/'
# file_names = ('non_polite_train', 'polite_train', 'non_polite_test', 'polite_test')
# file_paths = [f'{data_dir}/{fname}' for fname in file_names]
polite_train, polite_test, non_polite_train, non_polite_test, polite_vocab, non_polite_vocab, POLITE_PADDING_INDEX = get_data()
print("Preprocessing complete.")