import numpy as np

path1 = '../dataset/esm2_embeddings/A2ML1.npy'
data1 = np.load(path1)
print(len(data1))
print(data1)

print("------------------------------------------------------------------------------")
path2 = '../dataset/enformer_embeddings_hf/A2ML1.npy'
data2 = np.load(path2)
print(len(data2))
print(data2)

print("------------------------------------------------------------------------------")
path3 = '../dataset/orthrus_embeddings/orthrus_4track.pkl'
import pickle
with open(path3, 'rb') as f:
    data3 = pickle.load(f)
print(len(data3))
head = data3['A2ML1'][:5]
print(head)
print(data3['A2ML1'])
print(data3['A2ML1'].shape)
