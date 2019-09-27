import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal

x1, y1 = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j] 
x2, y2 = np.mgrid[-2.0:2.0:30j, -2.0:2.0:30j]
xy1 = np.column_stack([x1.flat, y1.flat])
xy2 = np.column_stack([x2.flat, y2.flat])

mu = np.array([0.0, 0.0])
sigma = np.array([.5, .5])
covariance = np.diag(sigma**2)

z1 = multivariate_normal.pdf(xy1, mean=mu, cov=covariance)
z1 = z1.reshape(x1.shape)


mu = np.array([0.0, 0.0])
sigma = np.array([1.0, 1.0])
covariance = np.diag(sigma**2)

z2 = multivariate_normal.pdf(xy2, mean=mu, cov=covariance)
z2 = z2.reshape(x2.shape)


fig = plt.figure()
ax = fig.add_subplot(111)


ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1,y1,z1, marker='x', color='b')
ax.scatter(x2,y2,z2, marker='o', color='r')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()