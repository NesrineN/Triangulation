#include "LinearLS.h"
#include <complex>
#include <cmath>
#include <iostream>
#include "libOrsa/libNumerics/matrix.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Triangulation {

    Vec solveSVD(const Mat& A, const Vec& b, double threshold) {
        int m = A.nrow();
        int n = A.ncol();

        // 1. Compute SVD: A = U * S * V^T
        Mat U(m, m);
        Vec S(n);
        Mat V(n, n);
        A.SVD(U, S, V);

        // 2. Compute S_inverse
        // We treat S as a diagonal matrix. For each s_i, if it's > threshold, we take 1/s_i
        Vec Sinv(n);
        for (int i = 0; i < n; ++i) {
            if (S(i) > threshold) {
                Sinv(i) = 1.0 / S(i);
            } else {
                Sinv(i) = 0.0;
            }
        }

        // 3. Solve x = V * Sinv * U^T * b
        // Step a: tmp = U^T * b
        Vec tmp = U.t() * b;

        // Step b: tmp2 = Sinv * tmp (element-wise multiplication since Sinv is diagonal)
        Vec tmp2(n);
        for (int i = 0; i < n; ++i) {
            tmp2(i) = Sinv(i) * tmp(i);
        }

        // Step c: x = V * tmp2
        Vec x = V * tmp2;

        return x;
    }

    // function that performs the triangulation using the Linear Least Squares method:
    // Here, we are setting X=(X,Y,Z,1) --> assuming point is not at infinity 
    // aim: to reduce set of equations to 4 non-homogeneous equations with 3 unknowns only

    Vec Triangulate_Linear_LS(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime){
        // extracting the coordinates from the u and u' vectors
        double u= U(0);
        double v= U(1);
        double u_p= U_prime(0);
        double v_p= U_prime(1);

        // creating the matrix A that is 4x4
        Mat A= Mat::zeros(4,4);

        Mat p0=P.copyRows(0,0);
        Mat p1=P.copyRows(1,1);
        Mat p2=P.copyRows(2, 2);

        Mat p0p=P_prime.copyRows(0,0);
        Mat p1p=P_prime.copyRows(1,1);
        Mat p2p=P_prime.copyRows(2, 2);

        // Row 0: uP3T-P1T
        Mat row0= u*p2 - p0;
        // Row 1: vP3T-P2T
        Mat row1= v*p2 - p1;
        // Row 2: u'P3'T - P1'T  
        Mat row2= u_p*p2p - p0p;
        // Row 3: v'P3'T - P2'T
        Mat row3= v_p*p2p - p1p;

        A.paste(0, 0, row0);
        A.paste(1,0, row1);
        A.paste(2,0, row2);
        A.paste(3,0, row3);

        // We separate into A' (first 3 columns) and b (negated 4th column)
        Mat A_prime = A.copyCols(0, 2);
        Vec b = -A.col(3);

        // We then Solve A'x = b using SVD pseudo-inverse method
        Vec solution=solveSVD(A_prime, b);

        // no dehomogenization needed cause we are in the non-homogeneous case
        if(solution(2)<0){
            std::cout << "Warning: Point triangulated behind camera (Z=" << solution(2) << ")" << std::endl;
            return Vec(0, 0, 0);
        }

        return solution;

    }


}





