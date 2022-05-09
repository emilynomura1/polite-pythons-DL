import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
#from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random

#from attenvis import AttentionVis
#av = AttentionVis()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(model, train_french, train_english, eng_padding_index, fr_padding_index):
	"""
	Runs through one epoch - all training examples.
	:param model: the initialized model to use for forward and backward pass
	:param train_french: French train data (all data for training) of shape (num_sentences, window_size)
	:param train_english: English train data (all data for training) of shape (num_sentences, window_size + 1)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""
	# NOTE: For each training step, you should pass in the French sentences to be used by the encoder,
	# and English sentences to be used by the decoder
	# - The English sentences passed to the decoder should have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
	#
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]

	# Format decoder inputs and decoder labels
	decoder_input = [i[:-1] for i in train_english]#train_english[:-1]
	decoder_label = [i[1:] for i in train_english]#train_english[1:]

	# train_batch=0

	# Batch training
	for i in range(0, len(train_french), model.batch_size): #batch training

		# train_batch+=1
		# print(train_batch)

		encoder_input_batch = train_french[i:i+model.batch_size]
		decoder_input_batch = decoder_input[i:i+model.batch_size]
		decoder_label_batch = decoder_label[i:i+model.batch_size]

		# Create mask within for-loop
		# mask = [i != eng_padding_index for i in decoder_label_batch]
		input_mask = [np.where(np.array(i)==fr_padding_index, False, True) for i in encoder_input_batch]
		mask = [np.where(np.array(i)==eng_padding_index, 0, 1) for i in decoder_label_batch]

		# Forward and backward pass
		with tf.GradientTape() as tape:
			probs = model.call(encoder_input_batch, decoder_input_batch, input_mask)
			loss = model.loss_function(probs, decoder_label_batch, mask)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#@av.test_func
def test(model, test_french, test_english, eng_padding_index, fr_padding_index):
	"""
	Runs through one epoch - all testing examples.
	:param model: the initialized model to use for forward and backward pass
	:param test_french: French test data (all data for testing) of shape (num_sentences, window_size)
	:param test_english: English test data (all data for testing) of shape (num_sentences, window_size + 1)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
	e.g. (my_perplexity, my_accuracy)
	"""
	# Note: Follow the same procedure as in train() to construct batches of data!

	decoder_input = [i[:-1] for i in test_english]
	decoder_label = [i[1:] for i in test_english]

	test_batch = 0
	test_loss = 0
	test_accuracy = 0
	num_valid_total = 0
	sum_acc = 0
	test_bleu = 0

	# Batch testing
	for i in range(0, len(test_french), model.batch_size):
		test_batch += 1
		# print(test_batch)

		encoder_input_batch = test_french[i:i+model.batch_size]
		decoder_input_batch = decoder_input[i:i+model.batch_size]
		decoder_label_batch = decoder_label[i:i+model.batch_size]

		# Create mask within for-loop
		# mask = [i != eng_padding_index for i in decoder_label_batch]
		
		input_mask = [np.where(np.array(i)==fr_padding_index, False, True) for i in encoder_input_batch]
		mask = [np.where(np.array(i)==eng_padding_index, 0, 1) for i in decoder_label_batch]

		num_valid_per_batch = tf.math.reduce_sum(mask) #get number of valid tokens (non-pad values) per batch
		num_valid_total += num_valid_per_batch #iteratively sum to get total number of valid tokens

		# Call model, compute loss & accuracy
		probs = model.call(encoder_input_batch, decoder_input_batch, input_mask)

		test_loss += model.loss_function(probs, decoder_label_batch, mask)
		test_accuracy = model.accuracy_function(probs, decoder_label_batch, mask)
		# test_bleu += model.bleu_score(probs, decoder_label_batch, mask)
		sum_acc += tf.math.multiply(test_accuracy, tf.cast(num_valid_per_batch, dtype=tf.float32))

	accuracy = tf.math.divide(sum_acc, tf.cast(num_valid_total, dtype=tf.float32))
	# bleu = tf.math.divide(test_bleu, tf.cast(num_valid_total, dtype=tf.float32)) #total number of correctly predicted words/total number of valid tokens
	perplexity = tf.math.exp(tf.math.divide(test_loss, tf.cast(num_valid_total, dtype=tf.float32))) #total loss/total number of valid tokens
	
	return perplexity, accuracy#, bleu

def translate(model, sentence, vocab_frn, vocab_eng, fr_padding_index, toword):
	# padded, ignore1 = pad_corpus(sentence, None)
	# tokenized = convert_to_id(vocab_frn, padded)
	sentence = tf.convert_to_tensor(sentence)
	sentence = tf.reshape(sentence, (1, len(sentence)))
	encoded, state1 = model.encoder_gru(tf.nn.embedding_lookup(model.input_embedding, sentence))
	decoder_sentence = tf.expand_dims([vocab_eng['*START*']], 0)
	result = []
	input_mask = [np.where(np.array(i)==fr_padding_index, False, True) for i in sentence]
	for i in range(model.english_window_size):
		#print(decoder_sentence)
		output_embeddings = tf.nn.embedding_lookup(model.output_embedding, decoder_sentence)
		gru2,state = model.decoder_gru(output_embeddings,initial_state = state1)
		query_mask = tf.ones(tf.shape(gru2)[:-1], dtype=bool)
		value_mask = tf.convert_to_tensor(input_mask, dtype=bool)
		att_output = model.additive_attention([gru2,encoded], mask=[query_mask, value_mask], training=True, return_attention_scores=False)
		dense1 = model.dense1(att_output)
		logits = model.densef(dense1)
		prbs = tf.nn.softmax(logits) # (1, 1, n)
		pred_id=np.random.choice(np.arange(prbs.shape[2]), p = np.asarray(prbs).flatten())
		# print(type(pred_id))
		# predicted_id = tf.argmax(prbs[0]).numpy()
		# print(type(prbs), prbs.shape)
		# print(type(predicted_id, predicted_id.shape))
		print(toword[pred_id])
		result = result+ [toword[pred_id]]
		if (toword[pred_id]=='*STOP*'):
			return result
		decoder_sentence = tf.expand_dims([pred_id], 0)
		#print(decoder_sentence)
	return result

def main():
	
	#model_types = {"RNN" : RNN_Seq2Seq, "TRANSFORMER" : Transformer_Seq2Seq}
	#if len(sys.argv) != 2 or sys.argv[1] not in model_types.keys():
		#print("USAGE: python assignment.py <Model Type>")
		#print("<Model Type>: [RNN/TRANSFORMER]")
		#exit()

	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	#if sys.argv[1] == "TRANSFORMER":
		#av.setup_visualization(enable=True)

	print("Running preprocessing...")
	#data_dir   = '../../data'
	#file_names = ('fls.txt', 'els.txt', 'flt.txt', 'elt.txt')
	#file_paths = [f'{data_dir}/{fname}' for fname in file_names]
	train_eng,test_eng, train_frn,test_frn, vocab_eng,vocab_frn,eng_padding_index,fr_padding_index = get_data()
	print("Preprocessing complete.")
	print(train_eng.shape, test_eng.shape, train_frn.shape, test_frn.shape)
	model = RNN_Seq2Seq(52, len(vocab_frn), 52, len(vocab_eng))

	# TODO:
	# Train and Test Model for 1 epoch.

	# Drop extra sentences since batch_size is not evenly divisible
	# train_eng = train_eng[:-55]
	# train_frn = train_frn[:-55]
	# test_eng = test_eng[:-25]
	# test_frn = test_frn[:-25]

	# Run on a subset of data
	train_eng = train_eng[0:50000]
	train_frn = train_frn[0:50000]
	test_eng = test_eng[0:1000]
	test_frn = test_frn[0:1000]

	print("train start")
	train(model, train_frn, train_eng, eng_padding_index, fr_padding_index)
	print("train finish")

	# Save trained model
	# model.save('../models/model2', save_format='tf')


	perplexity, accuracy = test(model, test_frn, test_eng, eng_padding_index, fr_padding_index)
	print("Perplexity: ", perplexity, "Accuracy: ", accuracy)

	sentence = test_frn[4]
	toword = {v: k for k, v in vocab_eng.items()}
	result = translate(model, sentence, vocab_frn, vocab_eng, fr_padding_index, toword)
	print(result)

	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above
	#av.show_atten_heatmap()

if __name__ == '__main__':
	main()