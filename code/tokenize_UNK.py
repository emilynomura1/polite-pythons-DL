from collections import Counter
from change_data import change_data

non_polite_train, polite_train, non_polite_test, polite_test = change_data()

# UNK NON_POLITE TRAIN
flat_list = [item for sublist in non_polite_train for item in sublist]
x = Counter(flat_list)
words_to_remove = dict((k,v) for k,v in x.items() if v==1)
# print(len(words_to_remove))
non_polite_train_omit_words = list(words_to_remove.keys())
# Actually replace words in non_polite_train with '*UNK*' if they appear in non_polite_train_omit_words
for sentence in non_polite_train:
    for word in sentence:
        if word in non_polite_train_omit_words:
            word='*UNK*'
print(non_polite_train[0:50])