import os
import zipfile

import numpy as np
import requests

url = "http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip"
outfile = "./rectangles_images.zip"

if not os.path.exists(outfile):
    print("Downloading %s..." % outfile)
    r = requests.get(url, stream=True)
    with open(outfile, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
else:
    print("%s already downloaded..." % outfile)

f = zipfile.ZipFile('rectangles_images.zip', 'r')
f.extractall('.')
f.close()

data = np.loadtxt('rectangles_im_train.amat')
X = data[:, :-1] / 1.0
Y = data[:, -1:]

data = np.loadtxt('rectangles_im_test.amat')
Xtest = data[:, :-1] / 1.0
Ytest = data[:, -1:]

np.savez('rectangles_im.npz', X=X, Y=Y, Xtest=Xtest, Ytest=Ytest)

# to load the data agqin, do:
# d = dict(np.load('rectangles_im.npz'))
