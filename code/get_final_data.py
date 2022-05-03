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

def get_final_data():
    non_polite_train = read_data('../tag-and-generate-data-prep/data/catcher/tokenized_non_polite_train')
    polite_train = read_data('../tag-and-generate-data-prep/data/catcher/tokenized_polite_train')
    non_polite_test = read_data('../tag-and-generate-data-prep/data/catcher/tokenized_non_polite_test')
    polite_test = read_data('../tag-and-generate-data-prep/data/catcher/tokenized_polite_test')

    return non_polite_train, polite_train, non_polite_test, polite_test