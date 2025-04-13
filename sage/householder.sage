def householder(basis, n):
    r = matrix(RR, basis)
    q = identity_matrix(RR, n)
    u = zero_vector(RR, n)
    
    for k in xsrange(n - 1):
        norm = 0
        for i in xsrange(k, n):
            norm += r[k][i] * r[k][i]
        norm = sqrt(norm)
        if norm < 1e-6:
            continue
        
        if r[k, k] >= 0:
            u[k] = r[k, k] + norm
        else:
            u[k] = r[k, k] - norm
        
        norm_u = u[k] * u[k]
        for i in xsrange(k + 1, n):
            u[i] = r[k][i]
            norm_u += u[i] * u[i]
        
        h = identity_matrix(RR, n)
        for i in xsrange(k, n):
            for j in xsrange(k, n):
                h[i, j] -= 2 * u[i] * u[j] / norm_u
        
        r = r * h
        q = h * q
    
    for i in xsrange(n):
        v = r[i, i]
        for j in xsrange(n):
            r[j, i] /= v
        for j in xsrange(n):
            q[i, j] *= v
    
    return r, q

b = random_matrix(ZZ, 5, 5)
r, q = householder(b, 5)
show(b)
show(householder(b, 5))
show(r * q)
q, r = b.gram_schmidt()
show(matrix(RR, r), matrix(RR, q))
