#include "IterativeLS.h"
#include "LinearLS.h"
#include <complex>
#include <cmath>
#include <iostream>
#include "libOrsa/libNumerics/matrix.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Triangulation {

// function that performs the triangulation using the Iterative LS Method:
// idea of weights: 
// multiply equations by 1/w where w = P3^T X
// start with w0=w'0=1
// solve X0 using Linear LS method
// update w1=P3^T X0 and w1'=P3'^T X0
// the first 2 rows of A get multiplied by 1/w1
// the last 2 rows of A get multiplied by 1/w1'
// re-solve X1 using Linear LS method
// and so on until Xi=Xi-1 --> convergence we stop and return Xi
// if more than 10 iterations and no convergence --> return (0,0,0) and we fall back in main to another method

Vec Triangulate_Iterative_LS(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime) {
    double u= U(0);
    double v= U(1);
    double u_p= U_prime(0);
    double v_p= U_prime(1);

    double w = 1.0, w_p = 1.0;
    Mat solution_1, solution_2;
    
    // Initial A matrix
    Mat A = Mat::zeros(4, 4);

    Mat p0=P.copyRows(0,0);
    Mat p1=P.copyRows(1,1);
    Mat p2=P.copyRows(2, 2); // 1x4

    Mat p0p=P_prime.copyRows(0,0);
    Mat p1p=P_prime.copyRows(1,1);
    Mat p2p=P_prime.copyRows(2, 2);

    for (int i = 0; i < 10; i++) {
        // Building A where we divide first 2 rows by the weight w and last 2 rows by the weight w'

        // Row 0: uP3T-P1T
        Mat row0= (u*p2 - p0)/w;
        // Row 1: vP3T-P2T
        Mat row1= (v*p2 - p1)/w;
        // Row 2: u'P3'T - P1'T  
        Mat row2= (u_p*p2p - p0p)/w_p;
        // Row 3: v'P3'T - P2'T
        Mat row3= (v_p*p2p - p1p)/w_p;

        A.paste(0, 0, row0);
        A.paste(1,0, row1);
        A.paste(2,0, row2);
        A.paste(3,0, row3);

        // LS:
        // We separate into A' (first 3 columns) and b (negated 4th column)
        // A.colRange(start, end) is exclusive of the end index
        Mat A_prime = A.copyCols(0, 2); // 4x3
        Vec b = -A.col(3);

        // We then Solve A'x = b using SVD pseudo-inverse method
        Vec solution=solveSVD(A_prime, b, 1e-9); // 3x1 
        solution_2=solution; // 3x1 matrix

        // Convergence check: checking if the new solution is different than the old solution
        if (!(solution_1.nrow() == 0 || solution_1.ncol() == 0)) {
            double diff = (solution_2.col(0) - solution_1.col(0)).qnorm();
            if (diff < 1e-6) break; // We converged!
        }

        // if we didnt converge we need to update the weights and continue
        solution_1 = solution_2; // solution_1 (and solution_2) is a matrix 3x1

        // We update the weights for next iteration
        // w = p20*X + P21*Y + P22*Z + P23
        // p2 is 1x4 
        Vec p2_xyz(3);
        p2_xyz(0) = p2(0, 0);
        p2_xyz(1) = p2(0, 1);
        p2_xyz(2) = p2(0, 2);

        w = dot(p2_xyz, solution_2.col(0)) + p2(0, 3);

        Vec p2p_xyz(3);
        p2p_xyz(0) = p2p(0, 0);
        p2p_xyz(1) = p2p(0, 1);
        p2p_xyz(2) = p2p(0, 2);

        w_p = dot(p2p_xyz, solution_2.col(0)) + p2p(0, 3);
        
        // to prevent division by zero
        if (std::abs(w) < 1e-9)
            w = std::copysign(1e-9, w);

        if (std::abs(w_p) < 1e-9)
            w_p = std::copysign(1e-9, w_p);
    }

    if(solution_2.nrow()<3) return Vec(0,0,0); // if there was a problem with svd

    if (solution_2(2,0) < 0) return Vec(0, 0, 0); // checking if Z is negative
    
    return solution_2.col(0);
    
}

}


