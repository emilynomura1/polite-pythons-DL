import ast

def read_data(file_name):
	"""
	DO NOT CHANGE
    Load text data from file
	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
    """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file:
			text.append(ast.literal_eval(line))
	return text

def get_final_data():
    non_polite_train = read_data('../tag-and-generate-data-prep/data/catcher/tokenized_non_polite_train_new.txt')
    polite_train = read_data('../tag-and-generate-data-prep/data/catcher/tokenized_polite_train_new.txt')
    non_polite_test = read_data('../tag-and-generate-data-prep/data/catcher/tokenized_non_polite_test_new.txt')
    polite_test = read_data('../tag-and-generate-data-prep/data/catcher/tokenized_polite_test_new.txt')

    return non_polite_train, polite_train, non_polite_test, polite_test
# non_polite_train, polite_train, non_polite_test, polite_test = get_final_data()
# print(non_polite_train[0:5])