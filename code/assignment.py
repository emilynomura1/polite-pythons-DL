from distutils.command.install_egg_info import to_filename
import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from rnn_model import RNN_Seq2Seq
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(model, non_polite_train, polite_train, polite_padding_index, non_polite_padding_index):
	"""
	Runs through one epoch - all training examples.
	:param model: the initialized model to use for forward and backward pass
	:param non_polite_train: Non-polite train data (all data for training) of shape (num_sentences, window_size)
	:param polite_train: Polite train data (all data for training) of shape (num_sentences, window_size + 1)
	:param polite_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:param non_polite_padding_index: the padding index, the id of the *PAD* token. This integer is used when masking attention.
	:returns: a list of losses with length = total number of batches
	"""

	# Format decoder inputs and decoder labels
	decoder_input = [i[:-1] for i in polite_train]
	decoder_label = [i[1:] for i in polite_train]
	loss_list = []

	# Batch training
	for i in range(0, len(non_polite_train), model.batch_size):

		encoder_input_batch = non_polite_train[i:i+model.batch_size]
		decoder_input_batch = decoder_input[i:i+model.batch_size]
		decoder_label_batch = decoder_label[i:i+model.batch_size]

		# Create mask within for-loop
		input_mask = [np.where(np.array(i)==non_polite_padding_index, False, True) for i in encoder_input_batch] #boolean mask
		mask = [np.where(np.array(i)==polite_padding_index, 0, 1) for i in decoder_label_batch] #binary mask

		# Forward and backward pass
		with tf.GradientTape() as tape:
			probs = model.call(encoder_input_batch, decoder_input_batch, input_mask)
			loss = model.loss_function(probs, decoder_label_batch, mask)
		gradients = tape.gradient(loss, model.trainable_variables)
		loss_list.append(loss)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	return loss_list

def test(model, non_polite_test, polite_test, polite_padding_index, non_polite_padding_index):
	"""
	Runs through one epoch - all testing examples.
	:param model: the initialized model to use for forward and backward pass
	:param non_polite_test: Non-polite test data (all data for testing) of shape (num_sentences, window_size)
	:param polite_test: Polite test data (all data for testing) of shape (num_sentences, window_size + 1)
	:param polite_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:param non_polite_padding_index: the padding index, the id of *PAD* token. This integer is used when masking attention.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
	"""

	decoder_input = [i[:-1] for i in polite_test]
	decoder_label = [i[1:] for i in polite_test]

	test_loss = 0
	test_accuracy = 0
	num_valid_total = 0
	sum_acc = 0
	test_bleu = 0

	# Batch testing
	for i in range(0, len(non_polite_test), model.batch_size):

		encoder_input_batch = non_polite_test[i:i+model.batch_size]
		decoder_input_batch = decoder_input[i:i+model.batch_size]
		decoder_label_batch = decoder_label[i:i+model.batch_size]

		# Create mask within for-loop
		input_mask = [np.where(np.array(i)==non_polite_padding_index, False, True) for i in encoder_input_batch]
		mask = [np.where(np.array(i)==polite_padding_index, 0, 1) for i in decoder_label_batch]

		num_valid_per_batch = tf.math.reduce_sum(mask) #get number of valid tokens (non-pad values) per batch
		num_valid_total += num_valid_per_batch #iteratively sum to get total number of valid tokens

		# Call model, compute loss & accuracy
		probs = model.call(encoder_input_batch, decoder_input_batch, input_mask)
		test_loss += model.loss_function(probs, decoder_label_batch, mask)
		test_accuracy = model.accuracy_function(probs, decoder_label_batch, mask)
		sum_acc += tf.math.multiply(test_accuracy, tf.cast(num_valid_per_batch, dtype=tf.float32))
		# test_bleu += model.bleu_score(probs, decoder_label_batch, mask)

	accuracy = tf.math.divide(sum_acc, tf.cast(num_valid_total, dtype=tf.float32))
	perplexity = tf.math.exp(tf.math.divide(test_loss, tf.cast(num_valid_total, dtype=tf.float32))) #total loss/total number of valid tokens
	# bleu = tf.math.divide(test_bleu, tf.cast(num_valid_total, dtype=tf.float32)) #total number of correctly predicted words/total number of valid tokens
	
	return perplexity, accuracy#, bleu

def translate(model, sentence, non_polite_vocab, polite_vocab, non_polite_padding_index, toword):
	# padded, ignore1 = pad_corpus(sentence, None)
	# tokenized = convert_to_id(vocab_frn, padded)
	sentence = tf.convert_to_tensor(sentence)
	sentence = tf.reshape(sentence, (1, len(sentence)))
	encoded, state1 = model.encoder_gru(tf.nn.embedding_lookup(model.input_embedding, sentence))
	decoder_sentence = tf.expand_dims([polite_vocab['*START*']], 0)
	result = []
	input_mask = [np.where(np.array(i)==non_polite_padding_index, False, True) for i in sentence]
	for i in range(POLITE_WINDOW_SIZE):
		#print(decoder_sentence)
		output_embeddings = tf.nn.embedding_lookup(model.output_embedding, decoder_sentence)
		gru2,state = model.decoder_gru(output_embeddings,initial_state = state1)
		query_mask = tf.ones(tf.shape(gru2)[:-1], dtype=bool)
		value_mask = tf.convert_to_tensor(input_mask, dtype=bool)
		att_output = model.additive_attention([gru2,encoded], mask=[query_mask, value_mask], training=True, return_attention_scores=False)
		dense1 = model.dense1(att_output)
		dense2 = model.dense2(dense1)
		dense3 = model.dense3(dense2)
		logits = model.densef(dense3)
		prbs = tf.nn.softmax(logits) # (1, 1, n)
		pred_id=np.random.choice(np.arange(prbs.shape[2]), p = np.asarray(prbs).flatten())
		# print(type(pred_id))
		# pred_id = tf.argmax(np.asarray(prbs).flatten()).numpy()
		# print(type(prbs), prbs.shape)
		# print(type(predicted_id, predicted_id.shape))
		#print(toword[pred_id])
		result = result+ [toword[pred_id]]
		if (toword[pred_id]=='*STOP*'):
			return result
		decoder_sentence = tf.expand_dims([pred_id], 0)
		#print(decoder_sentence)
	return result

def main():

	print("Running preprocessing...")
	polite_train, polite_test, non_polite_train, non_polite_test, polite_vocab, non_polite_vocab, polite_padding_index, non_polite_padding_index = get_data()
	print("Preprocessing complete.")
	# print(polite_train.shape, polite_test.shape, non_polite_train.shape, non_polite_test.shape)
	model = RNN_Seq2Seq(NON_POLITE_WINDOW_SIZE, len(non_polite_vocab), POLITE_WINDOW_SIZE, len(polite_vocab))

	# Drop extra sentences since batch_size is not evenly divisible
	polite_train = polite_train[:-55]
	non_polite_train = non_polite_train[:-55]
	polite_test = polite_test[:-25]
	non_polite_test = non_polite_test[:-25]

	# Run on a subset of data
	# polite_train = polite_train[0:5000]
	# non_polite_train = non_polite_train[0:5000]
	# polite_test = polite_test[0:1000]
	# non_polite_test = non_polite_test[0:1000]

	print("train start")
	loss_list = train(model, non_polite_train, polite_train, polite_padding_index, non_polite_padding_index)
	print("train finish")

	# Print plot of losses over each batch
	x = [i for i in range(len(loss_list))]
	plt.plot(x, loss_list)
	plt.title('Loss per batch')
	plt.xlabel('Batch')
	plt.ylabel('Loss')
	plt.show()

	# Save trained model
	# model.save('../models/model2', save_format='tf')
	

	perplexity, accuracy = test(model, non_polite_test, polite_test, polite_padding_index, non_polite_padding_index)
	print("Perplexity: ", perplexity, "Accuracy: ", accuracy)
	smooth = SmoothingFunction(epsilon = 1e-12).method1
	# Test translator on one input sentence
	#sentence = non_polite_test[5]
	toword = {v: k for k, v in polite_vocab.items()}
	bleu = []
	true_sentence = []
	pred_sentence = []
	to_translate = non_polite_test[0:1]
	p = polite_test[0:100]
	for i in range(len(p)):
		s = []
		sentence = p[i]
		non = to_translate[i]
		for word in sentence:
			if word != polite_padding_index:
				s.append(toword[word])
				print(word, toword[word])
		result = translate(model, non, non_polite_vocab, polite_vocab, non_polite_padding_index, toword)
		true_sentence.append(s)
		pred_sentence.append(result)
		bleu.append(sentence_bleu(s,result, smoothing_function = smooth))		
				

	
	print(np.mean(np.asarray(bleu)))
	print(true_sentence[10], pred_sentence[10])
	

if __name__ == '__main__':
	main()