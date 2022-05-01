# tr_length = []
# for i in non_polite_train:
#     tr_length.append(len(i))
# tr_length_arr = np.array(tr_length)
# train_ind = np.where(tr_length_arr<=50)[0]
# non_polite_train = np.asarray(non_polite_train)
# non_polite_train = non_polite_train[train_ind]
# polite_train = np.asarray(polite_train)
# polite_train = polite_train[train_ind]
# test_length = []
# for sentence in polite_train:
#     test_length.append(len(sentence))
# print(max(test_length))

# te_length = []
# for i in non_polite_test:
#     te_length.append(len(i))
# te_length_arr = np.array(te_length)
# test_ind = np.where(te_length_arr<=50)[0]
# non_polite_test = np.asarray(non_polite_test)
# non_polite_test = non_polite_test[test_ind]
# polite_test = np.asarray(polite_test)
# polite_test = polite_test[test_ind]
# test_length = []
# for sentence in polite_test:
#     test_length.append(len(sentence))
# print(max(test_length))