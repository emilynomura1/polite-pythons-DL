import numpy as np
import tensorflow as tf

class CNN_Seq2Seq(tf.keras.Model):
    def __init__(self, non_polite_window_size, non_polite_vocab_size, polite_window_size, polite_vocab_size):
        super(CNN_Seq2Seq, self).__init__()
        self.non_polite_vocab_size = non_polite_vocab_size
        self.polite_vocab_size = polite_vocab_size

        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 100
        
        self.non_polite_embedding_matrix =  tf.Variable(tf.random.normal([self.non_polite_vocab_size, self.embedding_size], mean=0, stddev=0.01, dtype=tf.float32))
        self.polite_embedding_matrix = tf.Variable(tf.random.normal([self.polite_vocab_size, self.embedding_size], mean=0, stddev=0.01, dtype=tf.float32))

		# Create positional encoder layers
        self.non_polite_positional_encoding = tf.keras.layers.Embedding(self.non_polite_window_size, self.embedding_size)
        self.polite_positional_encoding = tf.keras.layers.Embedding(self.polite_window_size, self.embedding_size)


        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters = 5, kernel_size = 2, strides = 1,padding = "same"),
            tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding = "same"),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv1D(filters = 5, kernel_size = 2, strides = 1,padding = "same"),
            tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding = "same"),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dropout(0.15),
        
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(filters = 5, kernel_size = , strides = 2),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Dense(),
            tf.keras.layers.Dropout(),
            tf.keras.layers.Conv1DTranspose(filters = 5, kernel_size = , strides = 2),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Dense(),
        ])
        self.final_dense = tf.keras.layers.Dense(self.polite_vocab_size, activation='softmax')

@tf.function
def call(self, encoder_input):
    encoder_output = self.encoder(encoder_input)
    decoder_output = self.decoder(encoder_output)
    probs = self.final_dense(decoder_output)
    return probs

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