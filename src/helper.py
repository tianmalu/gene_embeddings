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