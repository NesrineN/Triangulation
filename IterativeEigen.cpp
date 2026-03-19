#include "IterativeEigen.h"
#include "libOrsa/libNumerics/matrix.h"
#include <iostream>
#include <complex>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Triangulation {

// function that performs the triangulation using the Iterative Eigen Method:
// idea of weights: 
// multiply equations by 1/w where w = P3^T X
// start with w0=w'0=1
// solve X0 using SVD of matrix A
// update w1=P3^T X0 and w1'=P3'^T X0
// the first 2 rows of A get multiplied by 1/w1
// the last 2 rows of A get multiplied by 1/w1'
// re-solve X1 using triangulate_Linear_Eigen
// and so on until Xi=Xi-1 --> convergence we stop and return Xi
// if more than 10 iterations and no convergence --> return (0,0,0) and we fall back in main to another method

Vec Triangulate_Iterative_Eigen(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime) {
    double u= U(0);
    double v= U(1);
    double u_p= U_prime(0);
    double v_p= U_prime(1);

    double w = 1.0, w_p = 1.0;
    Mat solution_1, solution_2; 
    
    // Initial A matrix
    Mat A = Mat::zeros(4, 4);

    for (int i = 0; i < 10; i++) {
        // Building A where we divide first 2 rows by the weight w and last 2 rows by the weight w'

        Mat p0=P.copyRows(0,0);
        Mat p1=P.copyRows(1,1);
        Mat p2=P.copyRows(2, 2);

        Mat p0p=P_prime.copyRows(0,0);
        Mat p1p=P_prime.copyRows(1,1);
        Mat p2p=P_prime.copyRows(2, 2);

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

        Mat W(4,4);
        Mat V(4,4);
        Vec S(4);
        A.SVD(W,S,V);

        int minIndex = 0;
        for (int i = 1; i < 4; ++i)
            if (S(i) < S(minIndex))
                minIndex = i;

        solution_2 = V.col(minIndex); // solution_2 is a 4x1 matrix

        // Convergence check: checking if the new solution is different than the old solution
        if (!(solution_1.nrow() == 0 || solution_1.ncol() == 0)) {
            Mat difference = (solution_2-solution_1); // 4x1
            Vec difference_vec=difference.col(0);
            double diff=difference_vec.qnorm(); // distance between the 2 vectors 
            if (diff < 1e-6) break; // We converged!
        }

        // if we didnt converge we need to update the weights and continue
        solution_1 = solution_2; // solution_1 is a matrix 4x1

        // We update the weights for next iteration
        w = dot(p2.row(0), solution_1.col(0));
        w_p = dot(p2p.row(0), solution_1.col(0));
        
        // to prevent division by zero
        if (std::abs(w) < 1e-9) w = 1e-9;
        if (std::abs(w_p) < 1e-9) w_p = 1e-9;
    }

    // Final de-homogenization
    double w_hom = solution_2(3,0);
    if (std::abs(w_hom) > 1e-9) {
        double X = solution_2(0,0) / w_hom;
        double Y = solution_2(1,0) / w_hom;
        double Z = solution_2(2,0) / w_hom;
        return (Z < 0) ? Vec(0, 0, 0) : Vec(X, Y, Z);
    }

    return Vec(0, 0, 0);
}

}


