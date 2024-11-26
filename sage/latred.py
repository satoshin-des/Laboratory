import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as amt
import sys

def GenRandomBasis(n : int):
    """
    Generates random full-rank basis matrix.
    
    Parameters
    ----------
    n : int
        Number of rank.

    Returns
    -------
    b : numpy.ndarray
        random basis matrix(each basis is row-vector)
    """
    if n <= 1:
        print("Input Error: input rank is too few.")
        sys.exit(0)

    E = np.identity(n - 1)
    left = np.random.randint(1, 9999999, (n, 1))
    b = np.vstack([np.zeros(n - 1), E])
    b = np.hstack([left, b])
    return b.astype(int)


def Gram_Schmidt(b : np.ndarray):
    """
    Computes Gram-Schmidt orthogonal basis matrix.
    
    Parameters
    ----------
    b : numpy.ndarray
        A lattice basis matrix(Each basis is row vector).

    Returns
    -------
    GSOb : numpy.ndarray
        Gram-Schmidt orthogonal basis matrix of an input basis.
    mu : int
        Gram-Schmidt coefficient matrix
    """
    n, m = b.shape
    GSOb = np.zeros((n, m))
    mu = np.identity(n)

    for i in range(n):
        GSOb[i] = b[i].copy()
        for j in range(i):
            mu[i, j] = np.dot(b[i], GSOb[j]) / np.dot(GSOb[j], GSOb[j])
            GSOb[i] -= mu[i, j] * GSOb[j].copy()
    return GSOb, mu

def Gram_Schmidt_squared(b : np.ndarray):
    """
    Computes Gram-Schmidt orthogonal basis matrix.
    
    Parameters
    ----------
    b : numpy.ndarray
        A lattice basis matrix(Each basis is row vector).

    Returns
    -------
    B : numpy.ndarray
        Squared norm vector of each Gram-Schmidt orthogonal basis.
    mu : int
        Gram-Schmidt coefficient matrix
    """
    n, m = b.shape
    GSOb = np.zeros((n, m))
    mu = np.identity(n)
    B = np.zeros(n)

    for i in range(n):
        GSOb[i] = b[i].copy()
        for j in range(i):
            mu[i, j] = np.dot(b[i], GSOb[j]) / np.dot(GSOb[j], GSOb[j])
            GSOb[i] -= mu[i, j] * GSOb[j].copy()
        B[i] = np.dot(GSOb[i], GSOb[i])
    return B, mu


def SizeReduce(b : np.ndarray, mu : np.ndarray, i : int, j : int):
    n, m = b.shape

    if abs(mu[i, j]) > 0.5:
        q = round(mu[i, j])
        b[i] -= q * b[j].copy()
        mu[i][: j + 1] -= q * mu[j][: j + 1].copy()
    return b, mu


def LLLReduce(b : np.ndarray, d : float = 0.99):
    """
    LLL-reduces.
    
    Parameters
    ----------
    b : numpy.ndarray
        A lattice basis matrix(Each basis is row vector).
    d : float
        A reduction parameter.

    Returns
    -------
    b : numpy.ndarray
        LLL-reduced basis matrix.
    """
    n = b.shape[0]
    B, mu = Gram_Schmidt_squared(b)
    k = 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    images = []

    while k < n:
        
        #im = ax.plot(B, animated=True, color='blue')
        im = ax.bar(np.arange(len(B)) + 1, B, color='blue')
        plt.yscale('log')
        images.append(im)
        
        for j in range(k)[::-1]:
            if abs(mu[k, j]) > 0.5:
                q = round(mu[k, j])
                b[k] -= q * b[j].copy()
                mu[k][: j + 1] -= q * mu[j][: j + 1].copy()

        if B[k] < (d - mu[k, k - 1] * mu[k, k - 1]) * B[k - 1] and k > 0:
            b[k], b[k - 1] = b[k - 1], b[k].copy()

            #Update GSO-information
            nu = mu[k, k - 1]
            c = B[k] + nu * nu * B[k - 1]; c_inv = 1. / c
            mu[k, k - 1] = nu * B[k - 1] * c_inv
            B[k] *= B[k - 1] * c_inv
            B[k - 1] = c
            t = mu[k - 1, : k - 1].copy()
            mu[k - 1, : k - 1] = mu[k, : k - 1]
            mu[k, : k - 1] = t.copy()
            t = mu[k + 1 :, k].copy()
            mu[k + 1 :, k] = mu[k + 1 :, k - 1] - nu * t.copy()
            mu[k + 1 :, k - 1] = t.copy() + mu[k, k - 1] * mu[k + 1 :, k]
            k -= 1
        else: k += 1
    print("start")
    anime = amt.ArtistAnimation(fig, images, interval=100)
    anime.save("LLL.mp4", writer="ffmpeg")
    return b


def DeepLLLReduce(b : np.ndarray, d : float = 0.99):
    """
    Deep-LLL-reduces.
    
    Parameters
    ----------
    b : numpy.ndarray
        A lattice basis matrix(Each basis is row vector).
    d : float
        A reduction parameter.

    Returns
    -------
    b : numpy.ndarray
        Deep-LLL-reduced basis matrix.
    """
    B, mu = Gram_Schmidt_squared(b)
    k : int = 1
    n = len(b)
 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    images = []

    while k < n:
        print(k)
        im = ax.bar(np.arange(len(B)) + 1, B, color='blue')
        plt.yscale('log')
        images.append(im)
        
        for j in range(k)[::-1]:
            if mu[k, j] > 0.5 or mu[k, j] < -0.5:
                q = round(mu[k, j])
                b[k] -= q * np.copy(b[j])
                mu[k, : j + 1] -= q * np.copy(mu[j, : j + 1])

        C = np.dot(b[k], b[k])

        i = 0
        while i < k:
            if C >= d * B[i]:
                C -= mu[k, i] * mu[k, i] * B[i]
                i += 1
            else:
                v = np.copy(b[k])
                b[i + 1: k + 1] = np.copy(b[i: k])
                b[i] = np.copy(v)
                B, mu = Gram_Schmidt_squared(b)
                k = max(i - 1, 0)
        k += 1
    print("start")
    anime = amt.ArtistAnimation(fig, images, interval=100)
    anime.save("DeepLLL.mp4", writer="ffmpeg")
    return b


def PotLLLReduce(b, delta = 0.99):
    n, m = b.shape
    l = 0
    B, mu = Gram_Schmidt_squared(b)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    images = []
    
    while l < n:
        print(l)
        im = ax.bar(np.arange(len(B)) + 1, B, color='blue')
        plt.yscale('log')
        images.append(im)
        # Size-reduces the basis
        for j in range(l)[::-1]:
            if mu[l, j] > 0.5 or mu[l, j] < -0.5:
                q = round(mu[l, j])
                b[l] -= q * np.copy(b[j])
                mu[l, : j + 1] -= q * np.copy(mu[j, : j + 1])

        # Computes potential of the basis
        P = P_min = 1; k = 0
        for j in range(l)[::-1]:
            S = 0.0
            for i in range(j, l): S += mu[l, i] * mu[l, i] * B[i]
            P *= (B[l] + S) / B[j]
            if P < P_min:
                k = j; P_min = P

        # Deep-insertes the basis
        if delta > P_min:
            v = np.copy(b[l])
            b[k + 1: l + 1] = np.copy(b[k: l])
            b[k] = np.copy(v)
            B, mu = Gram_Schmidt_squared(b)

            l = k
        else:
            l += 1
    print("start")
    anime = amt.ArtistAnimation(fig, images, interval=100)
    anime.save("PotLLL.mp4", writer="ffmpeg")
    return b

rng = np.random.default_rng()
b = np.eye(50, dtype=int)
for i in range(50):
    b[i, 0] = rng.integers(100000, 1000000)
print(b)
#print(DeepLLLReduce(b))
#print(PotLLLReduce(b))
print(LLLReduce(b))