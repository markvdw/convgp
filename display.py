import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Display an opt_tools hist.')
parser.add_argument('hist_file', type=str, nargs='+', help="Paths to the history files to be displayed.")
parser.add_argument('--smooth', help="Smoothing for objective function.", type=int, default=3)
parser.add_argument('--logscale', help="Log scale graphs.", action="store_true")
args = parser.parse_args()

hs = [pd.read_pickle(hf) for hf in args.hist_file]

plt.figure()
for h, f in zip(hs, args.hist_file):
    fn = os.path.splitext(os.path.split(f)[-1])[0]
    plt.subplot(211)
    # plt.plot(h.t, np.convolve(h.f, np.ones(args.smooth) / args.smooth, 'same'), label=fn)
    plt.plot(h.t / 3600, h.f, label=fn)
    plt.subplot(212)
    plt.plot(h.t / 3600, h.learning_rate)
    plt.ylabel("Learning rate")
plt.subplot(211)
plt.legend()
plt.xlabel("Time (hrs)")
plt.ylabel("LML bound")
if args.logscale:
    plt.xscale('log')

# plt.figure()
# for h in hs:
#     # try:
#     f = h[~h.err.isnull()].filter(regex="model.kern.convrbf.basek*")
#     plt.plot(f)
#     # f = h.filter(regex="model.kern.convrbf.basek*")
#     # plt.plot(h.t, f[~f.acc.isnull()])
#     # except:
#     #     pass

# plt.figure()
# for i, h in enumerate(hs):
#     p = h[~h.acc.isnull()]
#     plt.plot(p.t / 3600, p.filter(regex="variance*"), color="C%i" % i)

plt.figure()
for h in hs:
    p = h[~h.acc.isnull()]
    plt.plot(p.t / 3600, p.err)
plt.xlabel("Time (hrs)")
plt.ylabel("Error")
if args.logscale:
    plt.xscale('log')

plt.figure()
for h in hs:
    p = h[~h.acc.isnull()]
    plt.plot(p.t / 3600, p.nlpp)
plt.xlabel("Time (hrs)")
plt.ylabel("nlpp")
if args.logscale:
    plt.xscale('log')


def reshape_patches_for_plot(patches):
    n_patch, dimx, dimy = patches.shape
    n_rows = int(np.floor(np.sqrt(n_patch)))
    n_cols = int(np.ceil(n_patch / n_rows))
    image = np.empty((n_rows * dimx + n_rows - 1, n_cols * dimy + n_cols - 1))
    image.fill(np.nan)
    for count, p in enumerate(patches):
        i = count // n_cols
        j = count % n_cols
        image[i * (dimx + 1):i * (dimx + 1) + dimx, j * (dimy + 1):j * (dimy + 1) + dimy] = p
    return image, n_rows, n_cols


patches_fig = plt.figure()
qm_fig = plt.figure()
w_fig = plt.figure()
for i, h in enumerate(hs):
    if not np.any(["conv" in cn for cn in h.columns]):
        continue

    nsbplt = int(np.ceil(len(hs) ** 0.5))

    plt.figure(patches_fig.number)
    plt.subplot(nsbplt, nsbplt, i + 1)
    Zend = h[~h.acc.isnull()].iloc[-1]['model.Z']
    patch_size = int(Zend.shape[-1] ** 0.5)

    qm = np.vstack(h[~h.acc.isnull()]['model.q_mu'].iloc[-1]).flatten()
    s = np.argsort(qm)

    patch_image, n_rows, n_cols = reshape_patches_for_plot(1 - Zend.reshape(-1, patch_size, patch_size)[s, :, :])
    plt.imshow(patch_image, cmap="gray")
    plt.clim(-0.25, 1.25)
    plt.colorbar()

    plt.figure(qm_fig.number)
    plt.subplot(nsbplt, nsbplt, i + 1)
    plt.imshow(np.hstack((qm[s], np.zeros(n_rows * n_cols - len(qm)))).reshape(n_rows, n_cols))
    plt.colorbar()

    plt.figure(w_fig.number)
    plt.subplot(nsbplt, nsbplt, i + 1)
    if "model.kern.weightedconv.W" in h.columns:
        plt.imshow(h[~h.acc.isnull()]["model.kern.weightedconv.W"].iloc[-1].reshape(26, 26))
        plt.colorbar()

plt.show()
