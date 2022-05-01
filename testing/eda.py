import numpy as np
import matplotlib.pyplot as plt

# Read in data
text = []
with open('../tag-and-generate-data-prep/data/catcher/polite_train', 'rt', encoding='latin') as data_file:
    for line in data_file: text.append(line.split())
#print(text[0:100])
#print(len(text))

length = []
for i in text:
    length.append(len(i))
#print(max(length))

#plt.hist(length)
#plt.show()

text_arr = np.array(text)
length_arr = np.array(length)
ind = np.where(length_arr<=50)
#print(ind)
subset = text_arr[np.where(length_arr<=25)]
print(max(length_arr))
#print(len(subset))

flat_list = [item for sublist in subset for item in sublist]
from collections import Counter
x = Counter(flat_list)
print(len(dict(x)))
y = dict((k,v) for k,v in x.items() if v>2)
print(y.keys())
#plt.hist(x.values)
#plt.show()

# DO UNK AND PADDING


# BUILD VOCAB
