import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Display an opt_tools hist.')
parser.add_argument('hist_file', type=str, nargs='+', help="Paths to the history files to be displayed.")
parser.add_argument('--smooth', help="Smoothing for objective function.", type=int, default=3)
parser.add_argument('--logscale', help="Log scale graphs.", action="store_true")
parser.add_argument('--xaxis', help="time|iter", type=str, default="time")
args = parser.parse_args()

hs = [pd.read_pickle(hf) for hf in args.hist_file]

plt.figure()
for h, f in zip(hs, args.hist_file):
    fn = os.path.splitext(os.path.split(f)[-1])[0]
    plt.subplot(211)
    idx = h.t / 3600 if args.xaxis == "time" else h.i
    # plt.plot(idx, np.convolve(h.f, np.ones(args.smooth) / args.smooth, 'same'), label=fn, alpha=0.5)
    smooth = np.ones(args.smooth) / args.smooth
    plt.plot(np.convolve(idx, smooth, 'valid'), np.convolve(h.f, smooth, 'valid'), label=fn, alpha=0.5)
    # plt.plot(idx, h.f, label=fn)
    plt.subplot(212)
    plt.plot(idx, h.learning_rate)
    plt.ylabel("Learning rate")
plt.subplot(211)
plt.legend()
plt.xlabel("Time (hrs)")
plt.ylabel("LML bound")
if args.logscale:
    plt.xscale('log')

plt.figure()
for h in hs:
    if 'lml' in h.columns:
        f = h[~(h.lml == 0.0) & ~np.isnan(h.lml)]
        plt.plot(f.t / 3600, f.lml, '-')

plt.figure()
for i, h in enumerate(hs):
    # try:
    # f = h[~h.err.isnull()].filter(regex="model.kern.convrbf.basek*")
    ss = h[~h.err.isnull()]
    f = ss.filter(regex=".*(lengthscales)")
    if f.shape[1] > 0:
        plt.plot(f, color="C%i" % i)
    f = ss.filter(regex=".*(variance)")
    plt.plot(f, color="C%i" % i, alpha=0.5)
    # f = h.filter(regex="model.kern.convrbf.basek*")
    # plt.plot(h.t, f[~f.acc.isnull()])
    # except:
    #     pass

# plt.figure()
# for i, h in enumerate(hs):
#     p = h[~h.acc.isnull()]
#     plt.plot(p.t / 3600, p.filter(regex="variance*"), color="C%i" % i)

plt.figure()
for h in hs:
    p = h[~h.acc.isnull()]
    plt.plot(p.t / 3600, p.err)
# plt.ylim(0.01, 0.03)
plt.xlabel("Time (hrs)")
plt.ylabel("Error")
if args.logscale:
    plt.xscale('log')

plt.figure()
for h in hs:
    p = h[~h.acc.isnull()]
    plt.plot(p.t / 3600, p.nlpp)
# plt.ylim(0.03, 0.08)
plt.xlabel("Time (hrs)")
plt.ylabel("nlpp")
if args.logscale:
    plt.xscale('log')

plt.figure()
for h in hs:
    plt.plot(h.t / h.tt)


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
    if not np.any(["conv" in cn or "basekern" in cn for cn in h.columns]):
        continue

    nsbplt = int(np.ceil(len(hs) ** 0.5))

    plt.figure(patches_fig.number)
    plt.subplot(nsbplt, nsbplt, i + 1)
    Zend = h[~h.acc.isnull()].iloc[-1]['model.Z' if 'model.Z' in h.columns else 'model.Z1']
    patch_size = int(Zend.shape[-1] ** 0.5)

    qmu = h[~h.acc.isnull()]['model.q_mu'].iloc[-1]
    if qmu.shape[1] == 1:
        qm = np.vstack(h[~h.acc.isnull()]['model.q_mu'].iloc[-1]).flatten()
        s = np.argsort(qm)
    else:
        s = np.arange(len(Zend))

    patch_image, n_rows, n_cols = reshape_patches_for_plot(1 - Zend.reshape(-1, patch_size, patch_size)[s, :, :])
    plt.imshow(patch_image, cmap="gray")
    plt.clim(-0.25, 1.25)
    plt.colorbar()

    if qmu.shape[1] == 1:
        plt.figure(qm_fig.number)
        plt.subplot(nsbplt, nsbplt, i + 1)
        plt.imshow(np.hstack((qm[s], np.zeros(n_rows * n_cols - len(qm)))).reshape(n_rows, n_cols))
        plt.colorbar()

    plt.figure(w_fig.number)
    plt.subplot(nsbplt, nsbplt, i + 1)
    Wseries = h[~h.acc.isnull()].iloc[-1].filter(regex=".*.W")
    if len(Wseries) >= 1:
        patch_weights = Wseries[0]
        patch_weights_img_size = int(np.ceil(patch_weights.shape[-1] ** 0.5))
        plt.imshow(patch_weights.reshape(patch_weights_img_size, patch_weights_img_size))
        plt.colorbar()

plt.show()
