import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as amt


def main():
    rng = np.random.default_rng()
    b = np.eye(50, dtype=int)
    for i in range(50):
        b[i, 0] = rng.integers(100000, 1000000)
    print(b)

    print(DeepLLLReduce(b))
    #print(PotLLLReduce(b))
    #print(LLLReduce(b))
    #print(LLLReduce_back(b))


def Gram_Schmidt_squared(b : np.ndarray):
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


def LLLReduce(b : np.ndarray, d : float = 0.99):
    n = b.shape[0]
    B, mu = Gram_Schmidt_squared(b)
    k = 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    images = []

    while k < n:
        im = ax.bar(np.arange(len(B)) + 1, np.log(B) / 2, color='blue')
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
    anime.save("LLL_nonlog.mp4", writer="ffmpeg")
    return b


def DeepLLLReduce(b : np.ndarray, d : float = 0.99):
    B, mu = Gram_Schmidt_squared(b)
    k : int = 1
    n = len(b)
 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(0, 14)
    images = []

    while k < n:
        print(k)
        im = ax.bar(np.arange(len(B)) + 1, np.log(B) / 2, color='blue')
        plt.ylim(0, 14)
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
    im = ax.bar(np.arange(len(B)) + 1, np.log(B) / 2, color='blue')
    plt.ylim(0, 14)
    images.append(im)
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
    plt.ylim(0, 14)
    images = []
    
    while l < n:
        print(l)
        im = ax.bar(np.arange(len(B)) + 1, np.log(B) / 2, color='blue')
        plt.ylim(0, 14)
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
    
    im = ax.bar(np.arange(len(B)) + 1, np.log(B) / 2, color='blue')
    plt.ylim(0, 14)
    images.append(im)
    
    print("start")
    anime = amt.ArtistAnimation(fig, images, interval=100)
    anime.save("PotLLL.mp4", writer="ffmpeg")
    return b


def LLLReduce_back(b, d : float = 0.99):
    n = b.shape[0]
    B, mu = Gram_Schmidt_squared(b)
    k = n - 2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    images = []
    
    while k >= 0:
        print(k)
        im = ax.bar(np.arange(len(B)) + 1, np.log(B) / 2, color='blue')
        images.append(im)
        for i in range(k, n):
            for j in range(i)[::-1]:
                if np.abs(mu[i, j]) > 0.5:
                    q = round(mu[i, j])
                    b[i] -= q * np.copy(b[j])
                    mu[i, : j + 1] -= q * np.copy(mu[j, : j + 1])

        if k <= n - 2 and B[k + 1] < (d - mu[k + 1, k] * mu[k + 1, k]) * B[k]:
            b[k + 1], b[k] = np.copy(b[k]), np.copy(b[k + 1])

            #Update GSO-information
            B, mu = Gram_Schmidt_squared(b)
            k += 1
        else: k -= 1
    
    im = ax.bar(np.arange(len(B)) + 1, np.log(B) / 2, color='blue')
    images.append(im)
    
    print("start")
    anime = amt.ArtistAnimation(fig, images, interval=100)
    anime.save("DualLLL.mp4", writer="ffmpeg")
    
    return b


    """
    void DualDeepLLLReduce(NTL::ZZ **b, const double d, const int n, const int m){
    int j, i, l;
    double **mu = (double **)malloc(n * sizeof(double *)), *B, **nu = (double **)malloc(n * sizeof(double *)), q, D;
    B = (double *)malloc(n * sizeof(double));
    for(i = 0; i < n; ++i){
        mu[i] = (double *)malloc(n * sizeof(double));
        nu[i] = (double *)malloc(n * sizeof(double));
    }
    GSO(b, B, mu, n, m);
    nu[n - 1][n - 1] = 1.0;
    
    for(int k = n - 2; k >= 0;){
        nu[k][k] = 1.0;

        // 部分双対サイズ基底簡約
        for(j = k + 1; j < n; ++j){
            /* 双対GSO係数 */
            nu[k][j] = 0;
            for(i = k; i < j; ++i) nu[k][j] -= mu[j][i] * nu[k][i];

            /* 双対サイズ基底簡約 */
            if(nu[k][j] > 0.5 || nu[k][j] < -0.5){
                q = round(nu[k][j]);
                for(i = 0; i < m; ++i) b[j][i] += q * b[k][i];
                for(i = j; i < n; ++i) nu[k][i] -= q * nu[j][i];
                for(i = 0; i <= k; ++i) mu[j][i] += q * mu[k][i];
            }
        }

        D = 0.0; l = n - 1;
        for(j = k; j < n; ++j) D += nu[k][j] * nu[k][j] / B[j];
        while(l > k){
            if(B[l] * D < d){
                DualDeepInsertion(b, m, k, l);
                k = fmin(l, n - 2) + 1;
                GSO(b, B, mu, n, m);
            }else{
                D -= nu[k][l] * nu[k][l] / B[l];
                --l;
            }
        }
        --k;
    }
}

    """


if __name__ == '__main__':
    main()
