import numpy as np

# Enumerates lattice vector whose norm is less than R
def ENUM(mu, B, R, n):
    sigma = zero_matrix(QQ, n + 1, n)
    r = np.roll(np.arange(n + 1) - 1, -1)
    rho = zero_vector(QQ, n + 1)
    v = zero_vector(ZZ, n) # Coefficient vector of the lattice vector
    center = sigma[0][:] # Standards of zigzag searching
    weight = v[:] # Weights of zigzag searching
    last_nonzero = k = 0
    v[0] = 1 # Avoids zero vector

    while True:
        # partial norm
        tmp = v[k] - center[k]; tmp *= tmp
        rho[k] = rho[k + 1] + tmp * B[k] # rho[k]=∥πₖ(v)∥

        if rho[k] <= R:
            if k == 0: # Finish searching all node
                return v, rho
            else:
                k -= 1
                r[k - 1] = max(r[k - 1], r[k])
                for i in xsrange(r[k], k, -1):
                    sigma[i, k] = sigma[i + 1, k] + mu[i, k] * v[i]
                center[k] = -sigma[k + 1, k]
                v[k] = round(center[k])
                weight[k] = 1
        else:
            k += 1
            if k == n:
                return None, None
            else:
                r[k - 1] = k
                if k >= last_nonzero:
                    last_nonzero = k
                    v[k] += 1
                else:
                    # Zigzag searching
                    if v[k] > center[k]: v[k] -= weight[k]
                    else: v[k] += weight[k]

                    # Updates weight of zigzag searching
                    weight[k] += 1


# Enumerates the shortest lattice vector
def EnumerateSV(mu, B, n):
    ENUM_v = zero_matrix(ZZ, n); R = B[0]
    D = vector(QQ, n + 1)
    while True:
        pre_ENUM_v = ENUM_v[:]
        DD = D # D[k]=∥πₖ(v)∥
        ENUM_v, D = ENUM(mu, B, R, n)
        if ENUM_v is None:
            return pre_ENUM_v, DD
        else:
            R = min(R, 0.99 * D[0])


def insert(b, x, n):
    alpha = QQ(0.99)
    beta = 4 / (4 * alpha - 1)
    U = zero_matrix(ZZ, n, n + 1)

    # Construction of gamma
    tmp = x.norm()
    tmp *= beta ^ ((n - 1) / 2)
    tmp += tmp
    gamma = round(tmp)

    # Construction of matrix
    for i in xsrange(n):
        U[i, i] = 1
        U[i, n] = gamma * x[i]
    U = U.LLL(delta=alpha)

    return U[:, : n] * b


# Computes Gram-Schmidt informations of dual-lattice
def DualGSO(b, n):
    B, mu = b.gram_schmidt()
    B = (B * B.T).diagonal()

    # dual GSO
    C = vector(QQ, n)
    hmu = identity_matrix(QQ, n, n)
    for i in xsrange(n):
        C[i] = 1 / B[i]
        for j in xsrange(i + 1, n):
            hmu[i, j] = -vector(mu[j, i: j]).inner_product(vector(hmu[i, i: j]))
    return C, hmu


def project_basis(k, l, b, n, m):
    GSOb, mu = b.gram_schmidt()
    pi_b = zero_matrix(QQ, l - k + 1, m)
    for i in xsrange(k, l + 1):
        for j in xsrange(k, n):
            pi_b[i - k] += (b[i] * GSOb[j]) / (GSOb[j] * GSOb[j]) * GSOb[j]
    return pi_b


def BKZ(b, beta, d, n, m):
    z = k = tour = 0
    while z < n - 1:
        # Goes to next tour
        if k == n - 1:
            k = 0
            tour += 1
            if tour >= 10:
                return b
        
        k1 = k; k += 1
        l = min(k1 + beta, n); h = min(l + 1, n)

        # Computes Gram-Schmidt informaion
        GSOb, mu = b.gram_schmidt()
        B = (GSOb * GSOb.T).diagonal()

        # Enumerates the shortest lattice vector on projection lattice L_{[k1, l]}
        w, D = EnumerateSV(mu[k1: l], B[k1: l], l - k1)
        s = w * project_basis(k1, l - 1, b, n, m)

        if ((not w.is_zero()) and B[k1] > s * s):
            z = 0

            # Construction of the lattice
            c = zero_matrix(ZZ, h + 1, m)
            c[: k1] = b[: k1]
            c[k1] = w * b[k1: l]
            c[k: h + 1] = b[k1: h]
            c = c.LLL(delta = d)

            b[: h] = c[1: h + 1]
        else:
            z += 1
            c = b[: h]
            b[: h] = c.LLL(delta = d)
    return b


def DualBKZ(b, beta, d, n, m):
    


# main
if __name__ == '__main__':
    n = ZZ(input())
    b = matrix(ZZ, n, n)

    for i in xsrange(n):
        b[i, i] = 1
        b[i, 0] = randint(9999, 99999)
    
    print(BKZ(b, 20, 0.99, n, n))
