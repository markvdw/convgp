import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib2tikz import save as tikz_save

plt.close('all')


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
    return image


df = pd.read_pickle("./results/rectangles-paper_hist.pkl")
df = df[~df.err.isnull()]
Zend = 1 - df.iloc[-1]['model.Z'].reshape(-1, 3, 3)
qm = np.vstack(df['model.q_mu'].iloc[-1]).flatten()
s = np.argsort(qm)
plt.imshow(reshape_patches_for_plot(Zend[s, :, :]), cmap="gray")
plt.clim([0.0, 1.1])
plt.axis('off')
plt.tight_layout()
plt.savefig("./tex/nips17/figures/rectangles.png")
print(df.err.iloc[-1])


df = pd.read_pickle("./results/rectangles-fullgp-rbf0_hist.pkl")
df = df[~df.err.isnull()]
print(df.err.iloc[-1])

plt.figure()
df = pd.read_pickle("./results/mnist01-conv50_hist.pkl")
Zend = df.iloc[-1]['model.Z']

qm = np.vstack(df['model.q_mu'].iloc[-1]).flatten()
s = np.argsort(qm)

plt.imshow(reshape_patches_for_plot(Zend.reshape(-1, 5, 5)[s, :, :]), cmap="gray")
plt.clim(-0.3, 1.0)
plt.axis('off')
plt.tight_layout()
plt.savefig("./tex/nips17/figures/conv01-patches.png")

plt.figure()
df = pd.read_pickle("./results/mnist01-wconv50_hist.pkl")
plt.imshow(df.iloc[-1]['model.kern.weightedconv.W'].reshape(24, 24))
# plt.clim(-0.3, 1.0)
plt.axis('off')
plt.tight_layout()
plt.savefig("./tex/nips17/figures/wconv01-W.png")


# Traces
dfs = [
    (pd.read_pickle('./results/fullmnist-rbf750_hist.pkl'), "RBF"),
    (pd.read_pickle('./results/fullmnist-conv750_hist.pkl'), "Conv - invariant"),
    (pd.read_pickle('./results/fullmnist-wconv750_hist.pkl'), "Conv - weighted"),
    (pd.read_pickle('./results/fullmnist-wconvrbf750-full_hist.pkl'), "Conv - weighted + RBF")
]

for df, descr in dfs:
    p = df[~df.err.isnull()]
    plt.figure(10)
    plt.plot(p.t / 3600, p.err * 100, label=descr)

    plt.figure(11)
    plt.plot(p.t / 3600, p.nlpp, label=descr)

    plt.figure(12)
    pss = p[~df.lml.isnull() & ~(df.lml == 0.0)]
    plt.plot(pss.t / 3600, pss.lml)

    print("%s\t acc: %f\tnlpp: %f" % (descr, p.iloc[-1].err, p.iloc[-1].nlpp))
plt.figure(10)
plt.xlabel("Time (hrs)")
plt.ylabel("Test error (\%)")
plt.xlim([0, 14])
plt.ylim(1, 3)
plt.savefig('./tex/nips17/figures/mnist-acc.png')
tikz_save('./tex/nips17/figures/mnist-acc.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)

plt.figure(11)
plt.xlabel("Time (hrs)")
plt.ylabel("Test nlpp")
plt.xlim([0, 14])
plt.ylim([0.035, 0.12])
plt.savefig('./tex/nips17/figures/mnist-nlpp.png')
tikz_save('./tex/nips17/figures/mnist-nlpp.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)

plt.figure(12)
plt.xlabel("Time (hrs)")
plt.ylabel("Evidence lower bound")
plt.xlim([0, 14])
plt.ylim([-30000, -6000])
plt.savefig('./tex/nips17/figures/mnist-bounds.png')
tikz_save('./tex/nips17/figures/mnist-bounds.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)

dfs = [(pd.read_pickle("./results/cifar10-rbf1000_hist.pkl"), "RBF"),
       (pd.read_pickle("./results/cifar10-wconv1000_hist.pkl"), "Conv - weighted"),
       (pd.read_pickle("./results/cifar10-cpwconv1000_hist.pkl"), "Colour conv - weighted"),
       (pd.read_pickle("./results/cifar10-addwconv1000_hist.pkl"), "Additive"),
       (pd.read_pickle("./results/cifar10-multi1000_hist.pkl"), "Multi-channel")]

for df, descr in dfs:
    p = df[~df.acc.isnull()]
    plt.figure(13)
    plt.plot(p.t / 3600, p.err * 100, label=descr)

    p = df[~df.acc.isnull()]
    plt.figure(15)
    plt.plot(p.t / 3600, p.nlpp, label=descr)

    plt.figure(14)
    pss = p.iloc[:(len(p) // 8) * 8]
    plt.plot(pss.t.reshape(-1, 8)[:, 0] / 3600, np.mean(pss.f.reshape(-1, 8), 1), label=descr)

plt.figure(13)
plt.xlim([0, 40])
plt.ylim([35, 60])
plt.xlabel("Time (hrs)")
plt.ylabel("Test error (\%)")
plt.savefig('./tex/nips17/figures/cifar-acc.png')
tikz_save('./tex/nips17/figures/cifar-acc.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)

plt.figure(14)
plt.xlim([0, 40])
plt.ylim([1e5, 2.5e5])
plt.xlabel("Time (hrs)")
plt.ylabel("Negative ELBO")
plt.savefig('./tex/nips17/figures/cifar-bounds.png')
tikz_save('./tex/nips17/figures/cifar-bounds.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)

plt.figure(15)
plt.xlim([0, 40])
# plt.ylim([1e5, 2.5e5])
plt.xlabel("Time (hrs)")
plt.ylabel("Test nlpp")
plt.savefig('./tex/nips17/figures/cifar-nlpp.png')
tikz_save('./tex/nips17/figures/cifar-nlpp.tikz', figurewidth="\\figurewidth",
          figureheight="\\figureheight", tex_relative_path_to_data="./figures/", show_info=False)
