import numpy as np

def read_data(file_name):
	"""
	DO NOT CHANGE
    Load text data from file
	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
    """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text

def change_data():

    # Read in data
    non_polite_train = read_data('../tag-and-generate-data-prep/data/catcher/non_polite_train')
    polite_train = read_data('../tag-and-generate-data-prep/data/catcher/polite_train')
    non_polite_test = read_data('../tag-and-generate-data-prep/data/catcher/non_polite_test')
    polite_test = read_data('../tag-and-generate-data-prep/data/catcher/polite_test')

    # Omit sentences in non-polite training and polite training
    # with length > 50 in POLITE TRAIN
    tr_length = []
    for i in polite_train:
        tr_length.append(len(i))
    tr_length_arr = np.array(tr_length)
    train_ind = np.where(tr_length_arr<=50)[0]
    non_polite_train = np.asarray(non_polite_train)
    non_polite_train = non_polite_train[train_ind]
    polite_train = np.asarray(polite_train)
    polite_train = polite_train[train_ind]

    # See the max sentence length in non-polite & polite training
    # non_polite_train_length = []
    # for sentence in non_polite_train:
    #     non_polite_train_length.append(len(sentence))
    # print(max(non_polite_train_length))
    # polite_train_length = []
    # for sentence in polite_train:
    #     polite_train_length.append(len(sentence))
    # print(max(polite_train_length))

    # Omit sentences in non-polite testing and polite testing
    # with length > 50 in POLITE TEST
    te_length = []
    for i in polite_test:
        te_length.append(len(i))
    te_length_arr = np.array(te_length)
    test_ind = np.where(te_length_arr<=50)[0]
    non_polite_test = np.asarray(non_polite_test)
    non_polite_test = non_polite_test[test_ind]
    polite_test = np.asarray(polite_test)
    polite_test = polite_test[test_ind]

    # See the max sentence length in non-polite & polite testing
    # non_polite_test_length = []
    # for sentence in non_polite_test:
    #     non_polite_test_length.append(len(sentence))
    # print(max(non_polite_test_length))
    # polite_test_length = []
    # for sentence in polite_test:
    #     polite_test_length.append(len(sentence))
    # print(max(polite_test_length))

    # print(type(non_polite_train[0:10]), type(polite_train[0:10]))
    return non_polite_train, polite_train, non_polite_test, polite_test