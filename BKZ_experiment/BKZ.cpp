#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <omp.h>

// External libraries
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <eigen3/Eigen/Dense>
#include <NTL/RR.h>
#include <NTL/vec_ZZ.h>
#include <NTL/mat_ZZ.h>
#include <NTL/LLL.h>

#include "BKZ.hpp"

static int BKZTour = 0;
static NTL::ZZ _;

void GSO(const MatrixXli b, VectorXld& B, MatrixXld& mu, const int n, const int m) {
    MatrixXld GSOb(n, m);

    for (int i = 0, j; i < n; ++i) {
        mu.coeffRef(i, i) = 1.0;
        GSOb.row(i) = b.row(i).cast<long double>();
        for (j = 0; j < i; ++j) {
            mu.coeffRef(i, j) = b.row(i).cast<long double>().dot(GSOb.row(j)) / GSOb.row(j).dot(GSOb.row(j));
            GSOb.row(i) -= mu.coeff(i, j) * GSOb.row(j);
        }
        B.coeffRef(i) = GSOb.row(i).dot(GSOb.row(i));
    }
}
void GSO(const MatrixXld b, VectorXld& B, MatrixXld& mu, const int n, const int m) {
    MatrixXld GSOb(n, m);

    for (int i = 0, j; i < n; ++i) {
        mu.coeffRef(i, i) = 1.0;
        GSOb.row(i) = b.row(i);
        for (j = 0; j < i; ++j) {
            mu.coeffRef(i, j) = b.row(i).dot(GSOb.row(j)) / GSOb.row(j).dot(GSOb.row(j));
            GSOb.row(i) -= mu.coeff(i, j) * GSOb.row(j);
        }
        B.coeffRef(i) = GSOb.row(i).dot(GSOb.row(i));
    }
}


long double logPot(const VectorXld B, const int n){
    NTL::RR logp; logp = 0;
    for (int i = 0; i < n; ++i) logp += (n - i) * NTL::log(NTL::to_RR((double)B.coeff(i)));
    return NTL::to_double(logp);
}


long double rho(const VectorXld B, const int n, const int m){
    long double S = 0, T = 0;
    for(int i = 0; i < n; ++i){
        S += (i + 1) * log(B.coeff(i));
        T += log(B.coeff(i));
    }
    return 12 * (S - 0.5 * (n + 1) * T) / (n * (n * n - 1));
}


long double SS(const VectorXld B, const int n){
    long double S = 0;
    for(int i = 0; i < n; ++i) S += B.coeff(i);
    return S;
}


long double OrthogonalituDefect(const MatrixXli b, const VectorXld B, const int n){
    long double P = 0;
    for(int i = 0; i < n; ++i) P += log(b.row(i).cast<long double>().norm()) - 0.5 * log(B.coeff(i));
    return P;
}


long double FrobNorm(const MatrixXli b){
    return sqrt((long double)(b.transpose() * b).trace());
}


long double TraceNorm(const MatrixXli b){
    return ((b.transpose() * b).cast<long double>().sqrt()).trace();
}


long double L_inftyNorm(const MatrixXli b, const int n, const int m){
    long double M = 0, S;
    for(int i = 0, j; i < n; ++i){
        S = 0.0;
        for(j = 0; j < n; ++j) S += abs(b.coeff(i, j));
        if(S >= M) M = S;
    }
    return M;
}


long double detlog(const MatrixXli b, const int n, const int m){
    long double S = 0;
    VectorXld B(n);
    MatrixXld mu(n, n), c = b.cast<long double>().log();
    GSO(c, B, mu, n, m);
    for(int i = 0; i < n; ++i) S += 0.5 * log(B.coeff(i));
    return S;
}


VectorXli ENUM(const MatrixXld mu, const VectorXld B, VectorXld& rho, const int n, const double R) {
    const int n1 = n + 1;
    int i, r[n1];
    double tmp;
    VectorXli w(n), v(n); w.setZero(); v.setZero(); v.coeffRef(0) = 1; // w: ジグザグに動く変分
    Eigen::VectorXd c(n); c.setZero(); rho.setZero();
    Eigen::MatrixXd sigma(n1, n); sigma.setZero();
    for (i = 0; i < n; ++i) r[i] = i;

    for (int k = 0, last_nonzero = 0; ;) {
        tmp = (double)v.coeff(k) - c.coeff(k); tmp *= tmp;
        rho.coeffRef(k) = rho.coeff(k + 1) + tmp * B.coeff(k); // rho[k]=∥πₖ(v)∥
        if (rho.coeff(k) <= R) {
            if (k == 0) return v;
            else{
                --k;
                r[k] = (r[k] > r[k + 1] ? r[k]: r[k + 1]);
                for (i = r[k]; i > k; --i) sigma.coeffRef(i, k) = sigma.coeff(i + 1, k) + mu.coeff(i, k) * v.coeff(i);
                c.coeffRef(k) = -sigma.coeff(k + 1, k);
                v.coeffRef(k) = round(c.coeff(k));
                w.coeffRef(k) = 1; // 小さいやつが見つかったら、変分を元に戻す
            }
        }
        else {
            ++k;
            if (k == n) {v.setZero(); return v;}
            else {
                r[k] = k;
                if (k >= last_nonzero) {
                    last_nonzero = k;
                    ++v.coeffRef(k);
                }
                else {
                    v.coeff(k) > c.coeff(k) ? v.coeffRef(k) -= w.coeff(k) : v.coeffRef(k) += w.coeff(k);
                    ++w.coeffRef(k);
                }
            }
        }
    }
}


VectorXli enumerate(const MatrixXld mu, const VectorXld B, VectorXld& rho, const int n) {
    VectorXli enum_v(n), pre_enum_v(n); enum_v.setZero(); pre_enum_v.setZero();
    VectorXld pre_rho(n + 1); pre_rho.setZero();
    for (double R = B.coeff(0);;) {
        pre_rho = rho; pre_enum_v = enum_v;
        enum_v = ENUM(mu, B, rho, n, R);
        if (enum_v.isZero()){
            rho = pre_rho;
            return pre_enum_v;
        }
        R *= 0.99;
    }
}


void __BKZ__(MatrixXli& b, const int beta, const double d, const int lp, const int n, const int m) {
    const int n1 = n - 1;
    FILE* fp = fopen("data/data.csv", "wt");
    VectorXli v, w;
    NTL::mat_ZZ c;
    VectorXld B(n), s; B.setZero();
    MatrixXld mu(n, n); mu.setIdentity();
    
    GSO(b, B, mu, n, m);
    fprintf(fp, "Potential,GSAslope,SS,FirstNorm,LastNorm,OrthogonalityDefect,Frobenius,TraceNorm,$L_{\\infty}$-norm,DetOfLog\n");

    for (int z = 0, j, t = 0, i, k = 0, h, lk1, l; z < n - 1;) {
        fprintf(fp, "%Lf,%Lf,%Lf,%lf,%lf,%Lf,%Lf,%Lf,%Lf,%Lf\n", logPot(B, n), -rho(B, n, m), SS(B, n), b.row(0).cast<double>().norm(), b.row(n - 1).cast<double>().norm(), OrthogonalituDefect(b, B, n), FrobNorm(b), TraceNorm(b), L_inftyNorm(b, n, m), -detlog(b, n, m));
        if(BKZTour >= lp) break;
        printf("z = %d\n", z);
        
        if (k == n1){k = 0; ++BKZTour;} ++k;
        l = std::min(k + beta - 1, n); h = std::min(l + 1, n);
        lk1 = l - k + 1;

        s.resize(lk1 + 1); s.setZero();
        w = enumerate(mu.block(k - 1, k - 1, lk1, lk1), B.segment(k - 1, lk1), s, lk1);
        if (B.coeff(k - 1) > s.coeff(k - 1) && (! w.isZero())) {
            z = 0;

            v = w * b.block(k - 1, 0, lk1, m);

            // Inserts and removes linearly-dependentness by LLL-reducing
            c.SetDims(h + 1, m);
            for(j = 0; j < m; ++j){
                for (i = 0; i < k - 1; ++i) c[i][j] = b.coeff(i, j);
                c[k - 1][j] = v.coeff(j);
                for (i = k; i < h + 1; ++i) c[i][j] = b.coeff(i - 1, j);
            }
            NTL::LLL(_, c, 99, 100);
            for(i = 1; i <= h; ++i) for(j = 0; j < m; ++j) b.coeffRef(i - 1, j) = NTL::to_long(c[i][j]);
            GSO(b, B, mu, n, m);
        }else{
            ++z;

            c.SetDims(h, m);
            for(j = 0; j < m; ++j) for(i = 0; i < h; ++i) c[i][j] = b.coeff(i, j);
            NTL::LLL(_, c, 99, 100);
            for(i = 0; i < h; ++i) for(j = 0; j < m; ++j) b.coeffRef(i, j) = NTL::to_long(c[i][j]);
            GSO(b, B, mu, n, m);
        }
    }
    fclose(fp);
}


extern "C" long **BKZ(long **b, const int beta, const double d, const int lp, const int n, const int m){
    int i, j;
    MatrixXli B(n, m);
    for(i = 0; i < n; ++i) for(j = 0; j < m; ++j) B.coeffRef(i, j) = b[i][j];
    __BKZ__(B, beta, d, lp, n, m);
    for(i = 0; i < n; ++i) for(j = 0; j < m; ++j) b[i][j] = B.coeff(i, j);
    return b;
}
