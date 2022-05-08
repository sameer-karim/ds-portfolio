import numpy as np

#scalars, vectors, matrices
# s = 5
# v = np.array([5,-2,4])
# m = np.array([[5,12,14],[3,-6,8]])

# #transpose
# print(m.T)

# print(type(s))
# print(type(v))
# print(type(m))

#tensor
m1 = np.array([[5,12,6],[-3,0,14]])
m2 = np.array([[9,8,7],[1,3,-5]])

t = np.array([m1,m2])

# print(t)
# print(m1.shape)

m3 = np.array([[5,12,6],[-3,0,14]])
m4 = np.array([[2,-1,6],[8,0,8]])

v = np.array([m3,m4])

print(t[0] + v[1])
