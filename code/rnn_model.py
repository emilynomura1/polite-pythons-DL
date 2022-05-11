import numpy as np
import tensorflow as tf
#from nltk.translate.bleu_score import sentence_bleu

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, non_polite_window_size, non_polite_vocab_size, polite_window_size, polite_vocab_size):

		super(RNN_Seq2Seq, self).__init__()
		self.non_polite_vocab_size = non_polite_vocab_size # The size of the non-polite vocab
		self.polite_vocab_size = polite_vocab_size # The size of the polite vocab

		self.non_polite_window_size = non_polite_window_size # The non-polite window size
		self.polite_window_size = polite_window_size # The polite window size

		# Define hyperparameters
		self.learning_rate = 0.001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		self.batch_size = 100
		self.embedding_size = 256 #maybe increase

		# Define embeddings, encoder, decoder, and feed forward layers
		def make_variables(*dims, initializer=tf.random.normal):
			return tf.Variable(initializer(dims, mean=0, stddev=0.01, dtype=tf.float32))
		self.input_embedding = make_variables(self.non_polite_vocab_size, self.embedding_size)
		self.output_embedding = make_variables(self.polite_vocab_size, self.embedding_size)

		self.encoder_gru = tf.keras.layers.GRU(200,return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
		self.decoder_gru = tf.keras.layers.GRU(200,return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

		self.additive_attention = tf.keras.layers.AdditiveAttention()
		self.dense1 = tf.keras.layers.Dense(self.embedding_size, activation='relu', use_bias=False) #change use_bias to True
		self.dense2 = tf.keras.layers.Dense(1000, activation='relu', use_bias=False)
		self.dense3 = tf.keras.layers.Dense(4000, activation='relu', use_bias=False) #maybe decrease; doesn't need to be this big
		self.densef = tf.keras.layers.Dense(self.polite_vocab_size)


	@tf.function
	def call(self, encoder_input, decoder_input, mask):
		"""
		:param encoder_input: batched ids corresponding to non-polite sentences
		:param decoder_input: batched ids corresponding to polite sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x polite_vocab_size]
        """

		input_embeddings = tf.nn.embedding_lookup(self.input_embedding, encoder_input)
		gru1, state1 = self.encoder_gru(input_embeddings, initial_state=None)
		#gru1b, state1b = self.encoder_gru(input_embeddings, initial_state=state1)
		output_embeddings = tf.nn.embedding_lookup(self.output_embedding, decoder_input)
		gru2, state2 = self.decoder_gru(output_embeddings, initial_state=state1)
		query_mask = tf.ones(tf.shape(gru2)[:-1], dtype=bool)
		value_mask = tf.convert_to_tensor(mask, dtype=bool)
		att_output = self.additive_attention([gru2,gru1], mask=[query_mask, value_mask], training=True, return_attention_scores=False)
		dense1 = self.dense1(att_output)
		dense2 = self.dense2(dense1)
		dense3 = self.dense3(dense2)
		logits = self.densef(dense3)
		prbs = tf.nn.softmax(logits)

		return prbs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE
		Computes the batch accuracy
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x polite_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.cast(tf.argmax(input=prbs, axis=2), dtype=tf.float32)
		labels = tf.cast(labels, dtype=tf.float32)
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

		loss = tf.math.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, from_logits=False), mask))
		return loss

	# def bleu_score(self, prbs, labels, mask):
	# 	"""
	# 	Calculates the bleu_score
	# 	"""
	# 	bleus = []
	# 	for i in range(len(labels)):
	# 		bleus += sentence_bleu(prbs[i:],labels[i])
		
	# 	return bleus/len(labels)


	# def decode_response(self, test_input):
	# 	#Getting the output states to pass into the decoder
	# 	states_value = se.call(test_input)
	# 	encoder_embed = tf.nn.embedding_lookup(self.fr_embedding_matrix, encoder_input)
	# 	output, state_h, state_c = self.encoder_rnn(encoder_embed)
	# 	encoder_final_state = [state_h, state_c]
		
	# 	#Generating empty target sequence of length 1
	# 	target_seq = np.zeros((1, 1, num_decoder_tokens))
	# 	#Setting the first token of target sequence with the start token
	# 	target_seq[0, 0, target_features_dict['<START>']] = 1.

	# 	#A variable to store our response word by word
	# 	decoded_sentence = ''

	# 	stop_condition = False
	# 	while not stop_condition:
	# 	#Predicting output tokens with probabilities and states
	# 	output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
	# 	#Choosing the one with highest probability
	# 	sampled_token_index = np.argmax(output_tokens[0, -1, :])
	# 	sampled_token = reverse_target_features_dict[sampled_token_index]
	# 	decoded_sentence += " " + sampled_token#Stop if hit max length or found the stop token
	# 	if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
	# 	stop_condition = True
	# 	#Update the target sequence
	# 	target_seq = np.zeros((1, 1, num_decoder_tokens))
	# 	target_seq[0, 0, sampled_token_index] = 1.
	# 	#Update states
	# 	states_value = [hidden_state, cell_state]
	# 	return decoded_sentence
