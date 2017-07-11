import zipfile
import numpy as np
import os

f = zipfile.ZipFile('rectangles.zip', 'r')
f.extractall('.')
f.close()

data = np.loadtxt('rectangles_train.amat')
X = data[:, :-1] / 1.0
Y = data[:, -1:]

data = np.loadtxt('rectangles_test.amat')
Xtest = data[:, :-1] / 1.0
Ytest = data[:, -1:]

np.savez_compressed('rectangles.npz', X=X, Y=Y, Xtest=Xtest, Ytest=Ytest)

os.remove('./rectangles_test.amat')
os.remove('./rectangles_train.amat')

# to load the data agqin, do:
# d = dict(np.load('rectangles.npz'))
