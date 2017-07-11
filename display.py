import argparse
import os

import matplotlib.pyplot as plt
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
#     try:
#         plt.plot(h.t, h.filter(regex="model.kern.ba*"))
#     except:
#         pass

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

plt.show()
