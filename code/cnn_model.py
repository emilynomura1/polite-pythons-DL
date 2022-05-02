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
        
        self.encoder = tf.keras.Sequential([
            tf.keras.layers,Conv1D(),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Dense(),
            tf.keras.layers.Dropout(),
            tf.keras.layers,Conv1D(),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Dense(),
            tf.keras.layers.Dropout(),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Dense(),
            tf.keras.layers.Dropout(),
            tf.keras.layers.Conv1DTranspose(),
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