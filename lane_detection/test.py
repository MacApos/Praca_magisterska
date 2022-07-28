import pickle
import numpy
import numpy as np

# data = pickle.load(open('data/data_array/160x80_data.p', 'rb'))
# labels = pickle.load(open('data/data_array/160x80_labels.p', 'rb'))
# coefficients = pickle.load(open('data/data_array/160x80_coefficients.p', 'rb'))
#
# data1 = pickle.load(open('data/data_array/Nowy folder/160x80_data.p', 'rb'))
# labels1 = pickle.load(open('data/data_array/Nowy folder/160x80_labels.p', 'rb'))
# coefficients1 = pickle.load(open('data/data_array/Nowy folder/160x80_coefficients.p', 'rb'))
#
# labels = np.array(labels)
# coefficients = np.array(coefficients)
#
# labels1 = np.array(labels1[:250])
# coefficients1 = np.array(coefficients1[:250])
# data1 = np.array(data1[:250])
#
# data_set = [labels, coefficients]
# data_set1 = [labels1, coefficients1]
#
# print(labels.shape, labels1.shape)
# print(np.array_equal(data, data1))
#
# for idx, (i, i1) in enumerate(zip(data_set, data_set1)):
#     print(np.array_equal(i, i1))

data = pickle.load(open('data/data_array/160x80_warp_data.p', 'rb'))
labels = pickle.load(open('data/data_array/160x80_warp_labels.p', 'rb'))
coefficients = pickle.load(open('data/data_array/160x80_warp_coefficients.p', 'rb'))

data1 = pickle.load(open('data/data_array/Nowy folder/160x80_warp_data.p', 'rb'))
labels1 = pickle.load(open('data/data_array/Nowy folder/160x80_warp_labels.p', 'rb'))
coefficients1 = pickle.load(open('data/data_array/Nowy folder/160x80_warp_coefficients.p', 'rb'))

labels = np.array(labels)
coefficients = np.array(coefficients)

labels1 = np.array(labels1[:250])
coefficients1 = np.array(coefficients1[:250])
data1 = np.array(data1[:250])

data_set = [labels, coefficients]
data_set1 = [labels1, coefficients1]

print(labels.shape, labels1.shape)
print(np.array_equal(data, data1))

for idx, (i, i1) in enumerate(zip(data_set, data_set1)):
    print(np.array_equal(i, i1))