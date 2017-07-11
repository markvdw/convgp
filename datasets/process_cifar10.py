import glob
import os
import pickle
import tarfile
import shutil

import numpy as np
import requests

url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
outfile = "cifar-10-python.tar.gz"

if not os.path.exists("./cifar-10-python.tar.gz"):
    print("Downloading %s..." % outfile)
    r = requests.get(url, stream=True)
    with open(outfile, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
else:
    print("%s already downloaded..." % outfile)

print("Unpacking...")
tar = tarfile.open(outfile, "r:gz")
tar.extractall()
tar.close()

images = None
labels = None
for fn in sorted(glob.glob("./cifar-10-batches-py/data_batch_*")):
    print(fn)
    with open(fn, 'rb') as f:
        l = pickle.load(f, encoding="bytes")
        images = np.vstack((images, l[b'data'])) if images is not None else l[b'data']
        labels = np.hstack((labels, l[b'labels'])) if labels is not None else l[b'labels']
labels = labels[:, None]

with open('./cifar-10-batches-py/test_batch', 'rb') as f:
    l = pickle.load(f, encoding="bytes")
    timages = l[b'data']
    tlabels = np.array(l[b'labels'])[:, None]

np.savez('cifar10.npz', X=images, Y=labels, Xt=timages, Yt=tlabels)

shutil.rmtree('./cifar-10-batches-py')
