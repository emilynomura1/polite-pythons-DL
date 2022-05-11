import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
# from transformer_model import Transformer_Seq2Seq
from basernn import BASE_RNN_Seq2Seq
import sys
import random
from matplotlib import pyplot as plt 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(model, train_non_polite, train_polite, polite_padding_index, non_polite_padding_index):
	"""
	Runs through one epoch - all training examples.
	:param model: the initialized model to use for forward and backward pass
	:param train_non_polite: Non polite train data (all data for training) of shape (num_sentences, window_size)
	:param train_polite: Polite train data (all data for training) of shape (num_sentences, window_size + 1)
	:param polite_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: loss_list: list of losses to plot in main
	"""
	# Format decoder inputs and decoder labels
	decoder_input = [i[:-1] for i in train_polite]
	decoder_label = [i[1:] for i in train_polite]
	
	loss_list = []
	# Batch training
	for i in range(0, len(train_non_polite), model.batch_size): 
		encoder_input_batch = train_non_polite[i:i+model.batch_size]
		decoder_input_batch = decoder_input[i:i+model.batch_size]
		decoder_label_batch = decoder_label[i:i+model.batch_size]

		# Create mask within for-loop
		input_mask = [np.where(np.array(i)==non_polite_padding_index, False, True) for i in encoder_input_batch]
		mask = [np.where(np.array(i)==polite_padding_index, 0, 1) for i in decoder_label_batch]

		# Forward and backward pass
		with tf.GradientTape() as tape:
			probs = model.call(encoder_input_batch, decoder_input_batch)
			loss = model.loss_function(probs, decoder_label_batch, mask)
		gradients = tape.gradient(loss, model.trainable_variables)
		loss_list.append(loss)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return loss_list


def test(model, test_non_polite, test_polite, polite_padding_index, non_polite_padding_index):
	"""
	Runs through one epoch - all testing examples.
	:param model: the initialized model to use for forward and backward pass
	:param test_non_polite: Non_polite test data (all data for testing) of shape (num_sentences, window_size)
	:param test_polite: Polite test data (all data for testing) of shape (num_sentences, window_size + 1)
	:param polite_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
	e.g. (my_perplexity, my_accuracy)
	"""
	decoder_input = [i[:-1] for i in test_polite]
	decoder_label = [i[1:] for i in test_polite]

	test_batch = 0
	test_loss = 0
	test_accuracy = 0
	num_valid_total = 0
	sum_acc = 0
	test_bleu = 0

	# Batch testing
	for i in range(0, len(test_non_polite), model.batch_size):
		test_batch += 1
		encoder_input_batch = test_non_polite[i:i+model.batch_size]
		decoder_input_batch = decoder_input[i:i+model.batch_size]
		decoder_label_batch = decoder_label[i:i+model.batch_size]

		# Create mask within for-loop
		input_mask = [np.where(np.array(i)==non_polite_padding_index, False, True) for i in encoder_input_batch]
		mask = [np.where(np.array(i)==polite_padding_index, 0, 1) for i in decoder_label_batch]

		num_valid_per_batch = tf.math.reduce_sum(mask) #get number of valid tokens (non-pad values) per batch
		num_valid_total += num_valid_per_batch # iteratively sum to get total number of valid tokens

		# Call model, compute loss & accuracy
		probs = model.call(encoder_input_batch, decoder_input_batch)

		test_loss += model.loss_function(probs, decoder_label_batch, mask)
		test_accuracy = model.accuracy_function(probs, decoder_label_batch, mask)
		
		sum_acc += tf.math.multiply(test_accuracy, tf.cast(num_valid_per_batch, dtype=tf.float32))

	accuracy = tf.math.divide(sum_acc, tf.cast(num_valid_total, dtype=tf.float32))
	perplexity = tf.math.exp(tf.math.divide(test_loss, tf.cast(num_valid_total, dtype=tf.float32))) 
	
	return perplexity, accuracy

def translate(model, sentence, vocab_non_polite, vocab_polite, non_polite_padding_index, toword):
	"""
	Translate a non_polite sentence by running it through the model and returns the output predicted polite sentence
	"""
	sentence = tf.convert_to_tensor(sentence)
	sentence = tf.reshape(sentence, (1, len(sentence)))

# 	# TRANSFORMER CODE 
# 	pos_emb_non_polite = model.position_non_polite.call(tf.nn.embedding_lookup(model.emb_non_polite, sentence))
# 	encoded = model.encoder(pos_emb_non_polite)
# 	decoder_input = tf.expand_dims([vocab_polite['*START*']], 0)
# 	result = []

# 	for i in range(model.polite_window_size):
# 		pos_emb_polite = model.position_polite.call(tf.nn.embedding_lookup(model.emb_polite, decoder_input))
# 		decoded = model.decoder(pos_emb_polite, encoded)
# 		prbs = model.D1(decoded)

 	# BASE RNN CODE 
	encoded, state1 = model.encoder(tf.nn.embedding_lookup(model.emb_non_polite, sentence))
	decoder_sentence = tf.expand_dims([vocab_polite['*START*']], 0)
	result = []
	input_mask = [np.where(np.array(i)==non_polite_padding_index, False, True) for i in sentence]
	for i in range(model.polite_window_size):
		
		output_embeddings = tf.nn.embedding_lookup(model.emb_polite, decoder_sentence)
		gru2,state = model.decoder(output_embeddings,initial_state = state1)
		prbs = model.D1(gru2)
		pred_id = np.random.choice(np.arange(prbs.shape[2]), p=np.asarray(prbs).flatten())
		
		if (toword[pred_id]=='*STOP*'):
			return result
		result = result+ [toword[pred_id]]
		decoder_sentence = tf.expand_dims([pred_id], 0)

	return result


def main():

	# PREPROCESS 
	print("Running preprocessing...")
	
	train_polite,test_polite, train_non_polite,test_non_polite, vocab_polite,vocab_non_polite,polite_padding_index,non_polite_padding_index = get_data()
	print("Preprocessing complete.")
	print(train_polite.shape, test_polite.shape, train_non_polite.shape, test_non_polite.shape)

	model = BASE_RNN_Seq2Seq(52, len(vocab_non_polite), 52, len(vocab_polite))
	# model = Transformer_Seq2Seq(52, len(vocab_non_polite), 52, len(vocab_polite))
	
	# TRAIN FOR 1 EPOCH

	# Drop extra sentences since batch_size is not evenly divisible
	# train_polite = train_polite[:-55]
	# train_non_polite = train_non_polite[:-55]
	# test_polite = test_polite[:-25]
	# test_non_polite = test_non_polite[:-25]

	# Run on a subset of data
	train_polite = train_polite[0:500]
	train_non_polite = train_non_polite[0:500]
	test_polite = test_polite[0:10]
	test_non_polite = test_non_polite[0:10]

	print("train start")
	loss_list = train(model, train_non_polite, train_polite, polite_padding_index, non_polite_padding_index)
	print("train finish")

	# Print plot of losses over each batch
	x = [i for i in range(len(loss_list))]
	plt.plot(x, loss_list)
	plt.title('Loss per batch')
	plt.xlabel('Batch')
	plt.ylabel('Loss')
	plt.show()

	# TEST 
	perplexity, accuracy = test(model, test_non_polite, test_polite, polite_padding_index, non_polite_padding_index)
	print("Perplexity: ", perplexity, "Accuracy: ", accuracy)

	# bleu score  
	toword = {v: k for k, v in vocab_polite.items()}
	smooth = SmoothingFunction(epsilon = 1e-12).method1
	bleu = []
	true_sentence = []
	pred_sentence = []
	to_translate = test_non_polite[0:100]
	p = test_polite[0:10]
	for i in range(len(p)):
		s = []
		sentence = p[i]
		non = to_translate[i]
		for word in sentence:
			if word != polite_padding_index:
				w = toword[word]
				if (w != "*START*" and w!="*STOP*"):
					s.append(toword[word])
        
		result = translate(model, non, vocab_non_polite, vocab_polite, non_polite_padding_index, toword)
		true_sentence.append(s)
		pred_sentence.append(result)
		bleu.append(sentence_bleu(s,result, smoothing_function = smooth))
	print(np.mean(np.asarray(bleu)))

	# print some example true and predicted sentence pairs 
	# print(true_sentence[10], pred_sentence[10])
	# print(true_sentence[27], pred_sentence[27])
	# print(true_sentence[33], pred_sentence[33])


if __name__ == '__main__':
	main()