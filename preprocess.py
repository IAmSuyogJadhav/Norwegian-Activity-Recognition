import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tsnecuda import TSNE as tsne
except ImportError:
    print('Please run install_tsnecuda.sh to install tsnecuda first (only if you want to use tsne_cuda).')

    
def save_nonzero(X_path="./raw-to-epoch/X.npy", Y_path="./raw-to-epoch/Y.npy", suffix='nonzero'):
    """
    Saves only the non-zero values separately in a numpy pickle.
    """
    X=np.load(X_path).squeeze()
    Y=np.load(Y_path)
    l = len(X) // 3
    
    X_x, X_y, X_z = X[l*0:l*1], X[l*1:l*2], X[l*2:l*3]
    Y_x, Y_y, Y_z = Y[l*0:l*1], Y[l*1:l*2], Y[l*2:l*3]
    nonzero_idx = ((Y_x != 0) | (Y_y != 0) | (Y_z != 0)).squeeze()
    print(f'{nonzero_idx.sum()} non zero values found.')

    X_x, X_y, X_z = X_x[nonzero_idx], X_y[nonzero_idx], X_z[nonzero_idx]
    Y_x, Y_y, Y_z = Y_x[nonzero_idx], Y_y[nonzero_idx], Y_z[nonzero_idx]

    X_cleaned = np.hstack([X_x.ravel()[..., None], X_y.ravel()[..., None], X_z.ravel()[..., None]])
    Y_cleaned = np.hstack([Y_x.ravel()[..., None], Y_y.ravel()[..., None], Y_z.ravel()[..., None]])
    
    np.save(f'./X_{suffix}.npy', X_cleaned)
    np.save(f'./Y_{suffix}.npy', Y_cleaned)
    print(f'Saved ./X_{suffix}.npy and ./Y_{suffix}.npy succesfully')
    

def save_xyz(X_path="./raw-to-epoch/X.npy", Y_path="./raw-to-epoch/Y.npy", suffix='3'):
    """
    Saves the X and Y arrays after reshaping them and adding an extra axis in the end (to store the cartesian x, y and z values in 3 different columns) in a numpy pickle.
    """
    X=np.load(X_path).squeeze()
    Y=np.load(Y_path)
    
    X = np.dstack([X[:int(len(X) / 3)][..., None], X[int(len(X) / 3):2*int(len(X) / 3)][..., None], X[2*int(len(X) / 3):][..., None]])
    Y = np.dstack([Y[:int(len(Y) / 3)][..., None], Y[int(len(Y) / 3):2*int(len(Y) / 3)][..., None], Y[2*int(len(Y) / 3):][..., None]])
    
    # Pad by additional two zero rows, to make it divisible by 7 (no. of days)
    X = np.vstack([X, np.zeros((2, 30, 3))])
    Y = np.vstack([Y, np.zeros((2, 3))])
    
    np.save(f'X_{suffix}.npy', X)
    np.save(f'Y_{suffix}.npy', Y)
    print(f'Saved ./X_{suffix}.npy and ./Y_{suffix}.npy succesfully')


def save_daywise(X_3_path='./X_3.npy', Y_3_path='./Y_3.npy', suffix='daywise'):
    """
    Saves the X_3 and Y_3 arrays after reshaping them and adding an extra axis in the start (to store each of the 7 days separately) in a numpy pickle.
    """
    X = np.load(X_3_path)
    Y = np.load(Y_3_path)
    t = 86400
    
    x_days = np.empty((7, t, 30, 3))
    x_days[0] = X[t*0:t*1]
    x_days[1] = X[t*1:t*2]
    x_days[2] = X[t*2:t*3]
    x_days[3] = X[t*3:t*4]
    x_days[4] = X[t*4:t*5]
    x_days[5] = X[t*5:t*6]
    x_days[6] = X[t*6:t*7]
    
    y_days = np.empty((7, t, 3))
    y_days[0] = Y[t*0:t*1]
    y_days[1] = Y[t*1:t*2]
    y_days[2] = Y[t*2:t*3]
    y_days[3] = Y[t*3:t*4]
    y_days[4] = Y[t*4:t*5]
    y_days[5] = Y[t*5:t*6]
    y_days[6] = Y[t*6:t*7]
    
    np.save(f'X_{suffix}.npy', x_days)
    np.save(f'Y_{suffix}.npy', y_days)
    print(f'Saved ./X_{suffix}.npy and ./Y_{suffix}.npy succesfully')


def fast_tsne(arr, n_components=2):
    # @article{chan2019gpu,
    #     title={GPU accelerated t-distributed stochastic neighbor embedding},
    #     author={Chan, David M and Rao, Roshan and Huang, Forrest and Canny, John F},
    #     journal={Journal of Parallel and Distributed Computing},
    #     volume={131},
    #     pages={1--13},
    #     year={2019},
    #     publisher={Elsevier}
    # }
    return tsne(n_components=n_components).fit_transform(arr)


def show_day(z, show=True):
    """
    Visualize the y data (should be a booolean array with labels activity (True) or no activity (False)) of a particular day.
    """
    act = plt.fill_between(np.arange(len(z)), z, cmap='Oranges')
    idle = plt.fill_between(np.arange(len(z)), np.invert(z), cmap='Greys')

    _ = plt.xticks(
        [i*3600 for i in range(25)],
        [str(hr).zfill(2) for hr in range(25)]
    )

    _ = plt.yticks([])
    plt.legend([act, idle], ['Activity', 'Idle'])
    plt.xlabel('Time of the day (HH)')
    
    # When used standalone
    if show:
        plt.show()


def show_days(z_daywise, thresh1=0, thresh0=50, size=86400, window=100, threshold=40, n_iters=1, savename=None):
    """
    Visualize the y data (should be a booolean array with labels activity (True) or no activity (False)) of a list of days.
    Some good combinations:
        Sharp Regions:
        show_days(Z_daywise, thresh1=80, window=500, threshold=300, n_iters=1)
    """
    n = len(z_daywise)
    plt.figure(figsize=(15, 2*n))
    for i, Z in enumerate(z_daywise):
        plt.subplot(n, 2, 2*i+1)
        plt.title(f'Original Day {i+1}')
        show_day(Z, show=False)
        
        ans = Z.copy()
        for _ in range(n_iters):
            ans = suppress_window(ans, size=size, window=window, threshold=threshold)
            ans = suppress_consecutive(ans, thresh1=thresh1, thresh0=thresh0)
        plt.subplot(n, 2, 2*i+2)
        plt.title(f'Enhanced Day {i+1}')
        show_day(ans, show=False)

        plt.tight_layout()
    
    if savename is not None:
        j = 1
        while os.path.exists(savename):
            print(f'\r{savename} already exists. Saving with a different name...', end='')       
            #  Add Unique number to the filename if the file already exists
            savename = f'{savename[:-4]}_{j}{savename[-4:]}'
            j += 1
        plt.savefig(savename, transparent=True)
    plt.show()


def suppress_consecutive(z, thresh1=0, thresh0=50):
    """
    Converts consecutive thresh1 or less no. of 1s to 0s.
    Converts consecutive thresh0 or less no. of 0s to 1s.
    """
    
    ans = z.copy()
    # For 0s
    if thresh0 != 0:
        df = pd.Series(np.invert(z))
        mapping = df.groupby((df != df.shift()).cumsum()).transform('size') * df < thresh0
        ans[mapping] = 1        
    
    # For 1s
    if thresh1 != 0:
        df = pd.Series(z)
        mapping = df.groupby((df != df.shift()).cumsum()).transform('size') * df < thresh1
        ans[mapping] = 0
    return ans


#0 filtering-basic
def suppress_window(temp_, size, window, threshold):
    """
    Converts a `window` of values to 1s if less than `threshold` amount of 0s are present.
    """
    temp = temp_.copy()
    for i in range(0,size,window):
        c0=0
        for j in range(min(window,size-i)):
            if(i+j<size and temp[i+j]==0):
                c0=c0+1
        if c0<=threshold:
            for j in range(min(window,size-i)):
                temp[i+j]=1
               
    return temp
