import ctypes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SDPB = ctypes.cdll.LoadLibrary("./libbkz.so")


def BKZ(b, beta, d):
    n, m = b.shape

    ptrs = [array.ctypes.data_as(ctypes.POINTER(ctypes.c_long)) for array in b]
    pp = (ctypes.POINTER(ctypes.c_long) * N)(*ptrs)

    SDPB.BKZ.argtypes = ctypes.POINTER(ctypes.POINTER(ctypes.c_long)), ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int
    SDPB.BKZ.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_long))
    bb = SDPB.BKZ(pp, beta, d, 10, n, m)

    for i in range(N):
        for j in range(N): b[i, j] = bb[i][j]

if __name__ == '__main__':
    if True:
        N = int(input())
        b = np.zeros((N, N), dtype=int)
        with open(f'svp_challenge/SVP-{N}-0.svp') as f: X = list(map(int, f.read().split()))
    
        for i in range(N):
            for j in range(N): b[i, j] = X[i * N + j + 1]

        c = b.copy()

        print(b)
        BKZ(c, 40, 0.99)
        print(np.linalg.norm(c[0]))
        print(c)
    data = pd.read_csv('data/data.csv')
    
    ddatas = data.columns.values

    for s in ddatas:
        fig, ax = plt.subplots()
        ax.set_xlabel("Tour")
        ax.set_ylabel(s)
    
        ax.plot(np.arange(len(data), dtype=int) / 100, data[s], marker="", label=s, color="red", lw=1.7)
    
        plt.tick_params()
        plt.legend()
        fig.set_size_inches(4 * 1.7, 3 * 1.7)
        plt.savefig(f'data/{s}.png')
