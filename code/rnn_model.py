import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
		###### DO NOT CHANGE ##############
		super(RNN_Seq2Seq, self).__init__()
		self.french_vocab_size = french_vocab_size # The size of the French vocab
		self.english_vocab_size = english_vocab_size # The size of the English vocab

		self.french_window_size = french_window_size # The French window size
		self.english_window_size = english_window_size # The English window size
		######^^^ DO NOT CHANGE ^^^##################

		# TODO:
		# 1) Define any hyperparameters
		self.learning_rate = 0.01
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		# Define batch size and optimizer/learning rate
		self.batch_size = 200 # You can change this
		self.embedding_size = 32 # You should change this

		# 2) Define embeddings, encoder, decoder, and feed forward layers
		def make_variables(*dims, initializer=tf.random.normal):
			return tf.Variable(initializer(dims, mean=0, stddev=0.01, dtype=tf.float32))
		self.fr_embedding_matrix = make_variables(self.french_vocab_size, self.embedding_size)
		self.eng_embedding_matrix = make_variables(self.english_vocab_size, self.embedding_size)
		self.encoder_rnn = tf.keras.layers.LSTM(self.embedding_size, return_sequences=True, return_state=True)
		self.decoder_rnn = tf.keras.layers.LSTM(self.embedding_size, return_sequences=True, return_state=True)
		self.dense1 = tf.keras.layers.Dense(60, activation=tf.nn.relu, use_bias=True) #80
		self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu, use_bias=True) #150
		self.dense3 = tf.keras.layers.Dense(self.english_vocab_size)
		self.dropout = tf.keras.layers.Dropout(rate=0.2)

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to French sentences
		:param decoder_input: batched ids corresponding to English sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		# TODO:
		# 1) Pass your French sentence embeddings to your encoder
		#https://keras.io/guides/working_with_rnns/

		encoder_embed = tf.nn.embedding_lookup(self.fr_embedding_matrix, encoder_input)
		output, state_h, state_c = self.encoder_rnn(encoder_embed)
		encoder_final_state = [state_h, state_c]

		# 2) Pass your English sentence embeddings, and final state of your encoder, to your decoder
		decoder_embed = tf.nn.embedding_lookup(self.eng_embedding_matrix, decoder_input)
		decoder_output, _, _ = self.decoder_rnn(decoder_embed, initial_state=encoder_final_state)

		# 3) Apply dense layer(s) to the decoder out to generate probabilities
		linear1 = self.dense1(decoder_output)
		linear1 = self.dropout(linear1)
		linear2 = self.dense2(linear1)
		linear2 = self.dropout(linear2)
		linear3 = self.dense3(linear2)
		prbs = tf.nn.softmax(linear3)
		# print(self.english_vocab_size, prbs.shape)

		return prbs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE
		Computes the batch accuracy
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		# decoded_symbols = tf.argmax(input=prbs, axis=2)
		decoded_symbols = tf.cast(tf.argmax(input=prbs, axis=2), dtype=tf.float32)
		labels = tf.cast(labels, dtype=tf.float32)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		loss = tf.math.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, from_logits=False), mask))
		return loss