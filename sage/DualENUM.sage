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

    while 1:
        # partial norm
        tmp = v[k] - center[k]; tmp *= tmp
        rho[k] = rho[k + 1] + tmp * B[k]

        if rho[k] <= R:
            if k == 0: # Finish searching all node
                return v, rho[k]
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
    while 1:
        pre_ENUM_v = ENUM_v[:]
        ENUM_v, _ = ENUM(mu, B, R, n)
        if ENUM_v is None:
            return pre_ENUM_v
        else:
            R = min(R, 0.99 * _)


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

# main
if __name__ == '__main__':
    n = 5
    b = matrix(ZZ, n, n)

    for i in xsrange(n):
        b[i, i] = randint(9999, 99999)
        b[i, 0] = 1
    
    b = b.LLL()
    #B, mu = b.gram_schmidt(); B = (B * B.T).diagonal()
    B, mu = DualGSO(b, n)

    print(EnumerateSV(mu, B, n))
