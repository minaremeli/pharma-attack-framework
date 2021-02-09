import scipy.io
import numpy as np
from os import path


def load_data(rel_path):
    X = scipy.io.mmread(path.join(rel_path, "T11_x.mtx")).tocsr()
    Y = scipy.io.mmread(path.join(rel_path, "T10_y.mtx")).tocsr()
    return X, Y


def load_partner_data(rel_path, partners_idx=[]):
    if not partners_idx:
        return
    X, Y = load_data(rel_path)
    partner_data = []
    for partner in partners_idx:
        rows_idx = np.genfromtxt(path.join(rel_path, "partner_%d_xy_row_index_map.csv" % partner),
                                 dtype=int,
                                 delimiter=',',
                                 skip_header=1,
                                 usecols=[1])
        cols_idx = np.genfromtxt(path.join(rel_path, "partner_%d_y_col_index_map.csv" % partner),
                                 dtype=int,
                                 delimiter=',',
                                 skip_header=1,
                                 usecols=[1])
        X_partner = X[rows_idx]
        Y_partner = Y[rows_idx].tocsc()[:, cols_idx].tocsr()

        print("Loaded %d. partner with %d samples and %d targets." % (partner, len(rows_idx), len(cols_idx)))

        partner_data.append((X_partner, Y_partner))

    return partner_data
