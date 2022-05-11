import numpy as np
import tensorflow as tf

class BASE_RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, non_polite_window_size, non_polite_vocab_size, polite_window_size, polite_vocab_size):
		
		super(BASE_RNN_Seq2Seq, self).__init__()
		self.non_polite_vocab_size = non_polite_vocab_size # The size of the non_polite vocab
		self.polite_vocab_size = polite_vocab_size # The size of the polite vocab

		self.non_polite_window_size = non_polite_window_size # The non_polite window size
		self.polite_window_size = polite_window_size # The polite window size
		
		# Define batch size and optimizer/learning rate
		self.batch_size = 100 
		self.embedding_size = 256 
		
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		# Define embeddings, encoder, decoder, and feed forward layers
		self.emb_non_polite = tf.Variable(tf.random.normal([self.non_polite_vocab_size, self.embedding_size], stddev=0.01))
		self.emb_polite = tf.Variable(tf.random.normal([self.polite_vocab_size, self.embedding_size], stddev=0.01))

		self.encoder = tf.keras.layers.GRU(40, return_sequences=True, return_state=True)
		self.decoder = tf.keras.layers.GRU(40, return_sequences=True, return_state=True)

		self.D1 = tf.keras.layers.Dense(self.polite_vocab_size , activation='softmax')


	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to non polite sentences
		:param decoder_input: batched ids corresponding to polite sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x polite_vocab_size]
		"""
		# Pass non_polite sentence embeddings to encoder
		embedding_non_polite = tf.nn.embedding_lookup(self.emb_non_polite, encoder_input)
		whole_output_encoder, final_state_encoder = self.encoder(embedding_non_polite, initial_state=None)  

		# Pass polite sentence embeddings, and final state of encoder to decoder
		embedding_polite = tf.nn.embedding_lookup(self.emb_polite, decoder_input)
		whole_output_decoder, final_state_decoder = self.decoder(embedding_polite, initial_state=final_state_encoder)  

		# Apply dense layer(s) to the decoder out to generate probabilities
		probabilities = self.D1(whole_output_decoder)   

		return probabilities

	def accuracy_function(self, prbs, labels, mask):
		"""
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x polite_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x polite_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		loss = tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs),tf.cast(mask,dtype=tf.float32))
		total_loss = tf.reduce_sum(loss)
		
		return total_loss
