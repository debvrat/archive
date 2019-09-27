import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('wine_data.csv').to_numpy()
data = data[:,1:]

#normalize the dataset
row, col = data.shape
for i in range(col):
	col_max = np.max(data[:,i])
	col_min = np.min(data[:,i])
	for j in range(row):
		data[j][i] = (data[j][i] - col_min)/(col_max - col_min)

covmat = np.cov(data, rowvar=False)
eigvals, eigvecs = np.linalg.eig(covmat)
#Coincidently, the eigvals vector returned above contains its highest values in the first two indices

eigvals_sort = eigvals[eigvals!=0] #non-zero elements
eigvals_sort = (eigvals_sort - np.min(eigvals_sort))/(np.max(eigvals_sort)-np.min(eigvals_sort)) #normalize
eigvals_sort = eigvals_sort[np.argsort(-eigvals_sort)] #sort

eigvals_sum = np.sum(eigvals_sort)
eigvals_len = len(eigvals_sort)

for k in range(1,eigvals_len+1,1):
	if (np.sum(eigvals_sort[:k])/eigvals_sum) >= 0.95:
		break
print('No. of components that can represent 95% of the variance = ' + str(k))

plt.figure()
plt.scatter(np.arange(1,eigvals_len+1,1),eigvals_sort)
plt.xlabel('index')
plt.ylabel('eigen value')
plt.title('eigen spectrum')

pc1 = eigvecs[:,0]
pc2 = eigvecs[:,1]

class1 = data[:59,]
class2 = data[58:129,]
class3 = data[129:177,]
plt.figure()
plt.scatter(np.dot(class1, pc1), np.dot(class1, pc2), cmap='red', label = 'class 1')
plt.scatter(np.dot(class2, pc1), np.dot(class2, pc2), cmap='green', label = 'class 2')
plt.scatter(np.dot(class3, pc1), np.dot(class3, pc2), cmap='blue', label = 'class 3')
plt.xlabel('pc 1')
plt.ylabel('pc 2')
plt.title('data projected on first two PCs')
plt.legend()

plt.show()



