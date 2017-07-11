import numpy as np


def thin_hist(df, target_per_decade):
    indices = np.unique(
        np.logspace(0, np.log10(df.i.max()), int(target_per_decade * np.log10(df.i.max()))).astype('int'))
    mat = indices[:, None] - df.i[None, :]
    ss = df.iloc[np.unique(np.argmin(np.abs(mat), 1)), :]
    return ss


if __name__ == "__main__":
    import argparse
    import os

    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="History file to thin.")
    parser.add_argument("--points", help="Number of points per decade.", type=int, default=10)
    parser.add_argument("--final", help="Store final only.", action="store_true")
    args = parser.parse_args()

    thick = pd.read_pickle(args.input)

    if not args.final:
        thin = thin_hist(thick, args.points)
    else:
        thin = thick.iloc[-1:, :]

    split = os.path.split(args.input)
    thin.to_pickle(os.path.join(*(split[:-1] + ("thin-" + split[-1],))))
