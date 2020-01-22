import numpy as np


def split_data(X, Y):
    assert (len(X) == len(Y)), "The lengths of X should be same as Y."
    assert len(X) % 3 == 0, "The no. of rows should be divisible by 3"

    l = len(X) // 3
    X_x, X_y, X_z = X[l*0:l*1], X[l*1:l*2], X[l*2:l*3]
    Y_x, Y_y, Y_z = Y[l*0:l*1], Y[l*1:l*2], Y[l*2:l*3]
    nonzero_idx = ((Y_x != 0) | (Y_y != 0) | (Y_z != 0)).squeeze()
    print(f'{nonzero_idx.sum()} non zero values found.')

    X_x, X_y, X_z = X_x[nonzero_idx], X_y[nonzero_idx], X_z[nonzero_idx]
    Y_x, Y_y, Y_z = Y_x[nonzero_idx], Y_y[nonzero_idx], Y_z[nonzero_idx]

    X_cleaned = np.hstack([X_x.ravel()[..., None], X_y.ravel()[..., None], X_z.ravel()[..., None]])
    Y_cleaned = np.hstack([Y_x.ravel()[..., None], Y_y.ravel()[..., None], Y_z.ravel()[..., None]])
    return X_cleaned, Y_cleaned

# Example Usage
# X = np.load("./raw-to-epoch/X.npy").squeeze()
# Y = np.load("./raw-to-epoch/Y.npy")

# X_cleaned, Y_cleaned = split_data(X, Y)
