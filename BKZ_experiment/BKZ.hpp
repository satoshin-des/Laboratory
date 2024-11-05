#ifndef LATTICE
#define LATTICE

#include <iostream>
#include <eigen3/Eigen/Dense>

typedef Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXli;
typedef Eigen::Matrix<long, 1, Eigen::Dynamic> VectorXli;
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXld;
typedef Eigen::Matrix<long double, 1, Eigen::Dynamic> VectorXld;

void GSO(const MatrixXli b, VectorXld& B, MatrixXld& mu, const int n, const int m);
void GSO(const MatrixXld b, VectorXld& B, MatrixXld& mu, const int n, const int m);
long double logPot(const VectorXld B, const int n);
long double rho(const VectorXld B, const int n, const int m);
long double SS(const VectorXld B, const int n);
long double OrthogonalituDefect(const MatrixXli b, const VectorXld B, const int n);
VectorXli ENUM(const MatrixXld mu, const VectorXld B, VectorXld& rho, const int n, const double R);
long double FrobNorm(const MatrixXli b);
VectorXli enumerate(const MatrixXld mu, const VectorXld B, VectorXld& rho, const int n);
void __BKZ__(MatrixXli& b, const int beta, const double d, const int lp, const int n, const int m);

extern "C" long **BKZ(long **b, const int beta, const double d, const int lp, const int n, const int m);

#endif // !LATTICE
