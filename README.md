# SugarCoding: Politeness Text Style Transfer
#### Polite Pythons: Pooja Barai, Emily Nomura, Megan Sindhi
This project attempts to solve a text style transfer problem - specifically politeness transfer, which entails transforming normal phrases/sentences into polite phrases/sentences. Politeness transfer can be helpful in industry-related or generally formal settings where sometimes academics or students with less experience tend to struggle with polite and formal communication. There has not been a lot of exploration in solving this problem, and the main paper that tackles this uses two sequence to sequence (seq2seq) models with transformers. One model tags where polite words should be and the other one generates the polite sentences. We wanted to see if we could simplify the style transfer by treating it as a machine translation problem and using one seq2seq model.
## Data and Preprocessing
The Politeness Dataset from the tag and generate paper by Madaan et al. is used. This dataset was formed from a subset of the Enron corpus which contains email exchanges from an American corporation. The original data did not consist of parallel corpuses, but the authors of the paper transformed the data and shared their preprocessing code, so we followed their directions on GitHub to generate parallel corpuses of non-polite and polite training and testing sets. Sentences greater than 50 words long were omitted in order to decrease training time and minimize the computational resources required to train the model. UNK tokens were substituted for words that appeared less than 20 times. This tokenization process decreased the size of the vocabulary and made the training data easier to manage. We then followed the general preprocessing structure of seq2seq models by padding sentences so they are all the same size, adding start and stop tokens, and converting sentences to vector IDs using the vocabulary dictionary.

Detailed instructions that outline how to acquire the dataset and generate parallel corpuses of training and testing data can be found in the original tag-and-generate README file located [here](https://github.com/emilynomura1/polite-pythons-DL/blob/master/tag-and-generate-README.md).

The tokenized data with a threshold level of 20 occurrences can be downloaded from [google drive](https://drive.google.com/drive/folders/1Md912QpVKOkr3i1HvTjb9tt9-Bqse3ci?usp=sharing).
## Model Architecture
The model architecture is a seq2seq model using an encoder, decoder, and additive attention layer. The encoder and decoder both contain an embedding layer and a gated recurrent unit (GRU). Both encoder and decoder GRUs have an output size of 200. A query mask and value mask are formed, and additive attention is calculated using the GRU outputs from the encoder and decoder. Three feed-forward layers are applied with a relu activation function, and a final dense layer is applied with a softmax activation in order to output the probabilities for each word in the vocabulary.
![Model Architecture](https://github.com/emilynomura1/polite-pythons-DL/blob/master/figures/architecture.png?raw=true)
### Evaluation Metrics
Per-symbol-accuracy, perplexity and bleu-score (BiLingual Evaluation Understudy) are calculated for the baseline and final model. All metrics are common evaluation metrics in the machine translation and style transfer realm.
## Results
Our final model using a RNN with attention achieved a perplexity of 4.95 and a per-symbol accuracy of 73%. Perplexity and per-symbol accuracy are compared to the baseline RNN model in the table below. The loss per batch for the RNN with attention decreases steadily from 18,000 to around 400.
![Evaluation Table](https://github.com/emilynomura1/polite-pythons-DL/blob/master/figures/eval-table.png?raw=true)
![Loss Per Batch](https://github.com/emilynomura1/polite-pythons-DL/blob/master/figures/finalrnnattention.png?raw=true)
### Comparison with Baseline
The RNN with attention is compared to a baseline RNN meant for simple machine translation tasks. The baseline RNN is composed of an encoder embedding, decoder embedding, two gated recurrent units (GRUs) with an output size of 40 for the encoder and decoder, and a single feed-forward layer with an output size equal to the polite vocabulary size. An adam optimizer with a learning rate of 1e-3 was used along with a batch size of 100 and an embedding size of 256. 

The baseline RNN model achieved a perplexity of 42 and a per-symbol accuracy of 38.5%. The loss decreased steadily for each batch, but failed to reach a loss of under 8000. Thus, compared to a baseline RNN model, the RNN with attention achieved a significantly lower perplexity and higher per-symbol accuracy.
## Challenges
The process of implementing our model ran relatively smoothly since there were many online resources for machine translation using RNNs. The model structure followed a general seq2seq architecture with an encoder and decoder. However, it was a bit unclear how to generate the attention masks when implementing additive attention.

Another challenge that arose was training time, which originally took around 2 hours to run on the department computer. After increasing the threshold of occurrences necessary in order to replace a certain word with an “UNK” token, the vocabulary size decreased from around thirty-one thousand to eight thousand, which led to a much more palatable training time of 35 minutes on a department machine.

We struggled the most with actually translating sentences which was necessary in order to test how well the model transformed an input sentence to a polite sentence and to calculate the bleu score. The translator did not seem to be able to correctly match the length of the true sentence and frequently predicted 50 words even if the true sentence was significantly shorter. Additionally, while the translator kept some of the content and added politeness, it seemed to simply repeat words and coult not create a grammatically correct output sentence.

The root cause of the bad translations is likely due to the limitations of the dataset itself. Since the data was not originally parallel, the parallel corpuses had to be generated by machine. The generated non-polite sentences are not very coherent, so the machine may have had issues matching the context and meaning of the non-polite sentences to the polite sentences.
