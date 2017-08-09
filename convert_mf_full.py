"""
convert_mf_full.py
Convert a hist pickle containing the history of a MeanFieldSVSumGP to a FullSVSumGP.
"""

import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Convert the history of a MeanFieldSVSumGP to a FullSVSumGP.')
parser.add_argument('hist_file', type=str, help="Paths to the history of the MeanFieldSVSumGP run.")
parser.add_argument('--output', '-o', default=None, type=str, help="Paths to the history of the MeanFieldSVSumGP run.")
args = parser.parse_args()

mf = pd.read_pickle(args.hist_file)
param_idx = mf[~mf['model.q_mu'].isnull()].index

q_mu = mf.loc[param_idx]['model.q_mu'].item()
M = q_mu.shape[0]
num_latent = q_mu.shape[1] // 2
q_mu = q_mu.reshape(M, num_latent, 2)
q_sqrt = mf.loc[param_idx]['model.q_sqrt'].item()
q_sqrt = q_sqrt.reshape(M, M, num_latent, 2)

fq_mu = q_mu.transpose([2, 0, 1]).reshape(2 * M, num_latent)

fq_sqrt = np.zeros((M * 2, M * 2, num_latent))
fq_sqrt[:M, :M, :] = q_sqrt[:, :, :, 0]
fq_sqrt[M:, M:, :] = q_sqrt[:, :, :, 1]

mf.set_value(param_idx, 'model.q_mu', [fq_mu])
mf.set_value(param_idx, 'model.q_sqrt', [fq_sqrt])

if args.output is None:
    if '-mf' in args.hist_file:
        output_file = args.hist_file.replace('-mf', '-full-converted')
    else:
        input_name, ext = os.path.splitext(args.hist_file)
        output_file = input_name + "-full-converted" + ext
else:
    output_file = args.output
mf.to_pickle(output_file)
