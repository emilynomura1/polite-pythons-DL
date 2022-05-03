import numpy as np
from collections import Counter
from change_data import change_data

non_polite_train, polite_train, non_polite_test, polite_test = change_data()

# UNK NON_POLITE TRAIN
flat_list = [item for sublist in non_polite_train for item in sublist]
x = Counter(flat_list)
words_to_remove = dict((k,v) for k,v in x.items() if v==1)
# print(len(words_to_remove))
omit_words = list(words_to_remove.keys())
# print(non_polite_train_omit_words[0:100])
# Actually replace words in non_polite_train with '*UNK*' if they appear in non_polite_train_omit_words
for i, sentence in np.ndenumerate(non_polite_train):
    for j, word in enumerate(sentence):
        if word in omit_words:
            non_polite_train[i][j] = '*UNK*'
# print(non_polite_train.tolist())

# UNK POLITE TRAIN
# flat_list = [item for sublist in polite_train for item in sublist]
# x = Counter(flat_list)
# words_to_remove = dict((k,v) for k,v in x.items() if v==1)
# omit_words = list(words_to_remove.keys())
# for i, sentence in np.ndenumerate(polite_train):
#     for j, word in enumerate(sentence):
#         if word in omit_words:
#             polite_train[i][j] = '*UNK*'

# UNK NON_POLITE TEST
# flat_list = [item for sublist in non_polite_test for item in sublist]
# x = Counter(flat_list)
# words_to_remove = dict((k,v) for k,v in x.items() if v==1)
# omit_words = list(words_to_remove.keys())
# for i, sentence in np.ndenumerate(non_polite_test):
#     for j, word in enumerate(sentence):
#         if word in omit_words:
#             non_polite_test[i][j] = '*UNK*'

# UNK POLITE TEST
# flat_list = [item for sublist in polite_test for item in sublist]
# x = Counter(flat_list)
# words_to_remove = dict((k,v) for k,v in x.items() if v==1)
# omit_words = list(words_to_remove.keys())
# for i, sentence in np.ndenumerate(polite_test):
#     for j, word in enumerate(sentence):
#         if word in omit_words:
#             polite_test[i][j] = '*UNK*'

# SAVE 4 NEW FILES
# file1 = open('../tag-and-generate-data-prep/data/catcher/tokenized_non_polite_train.txt', "w+")
# file1.write(str(non_polite_train))
# file1.close()
