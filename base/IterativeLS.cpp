#include "IterativeLS.h"
#include <complex>
#include <opencv2/opencv.hpp>

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

cv::Point3d triangulate_Iterative_LS(const cv::Vec3d& U, const cv::Vec3d& U_prime, const cv::Mat& P, const cv::Mat& P_prime) {
    double u = U[0], v = U[1];
    double u_p = U_prime[0];
    double v_p = U_prime[1];

    double w = 1.0, w_p = 1.0;
    cv::Mat solution_1, solution_2;
    
    // Initial A matrix
    cv::Mat A = cv::Mat::zeros(4, 4, CV_64F);

    for (int i = 0; i < 10; i++) {
        // Building A where we divide first 2 rows by the weight w and last 2 rows by the weight w'
        A.row(0) = (u * P.row(2) - P.row(0)) / w;
        A.row(1) = (v * P.row(2) - P.row(1)) / w;
        A.row(2) = (u_p * P_prime.row(2) - P_prime.row(0)) / w_p;
        A.row(3) = (v_p * P_prime.row(2) - P_prime.row(1)) / w_p;

        // LS:
        // We separate into A' (first 3 columns) and b (negated 4th column)
        // A.colRange(start, end) is exclusive of the end index
        cv::Mat A_prime = A.colRange(0, 3); 
        cv::Mat b = -A.col(3);

        // We then Solve A'x = b using SVD pseudo-inverse method
        cv::Mat solution_2;
        bool success = cv::solve(A_prime, b, solution_2, cv::DECOMP_SVD);

        if (!success) {return cv::Point3d(0, 0, 0);}

        // Convergence check: checking if the new solution is different than the old solution
        if (!solution_1.empty()) {
            double diff = cv::norm(solution_1, solution_2, cv::NORM_L2);
            if (diff < 1e-6) break; // We converged!
        }

        // if we didnt converge we need to update the weights and continue
        solution_1 = solution_2.clone();

        // We update the weights for next iteration
        w = P.row(2).dot(solution_1);
        w_p = P_prime.row(2).dot(solution_1);
        
        // to prevent division by zero
        if (std::abs(w) < 1e-9) w = 1e-9;
        if (std::abs(w_p) < 1e-9) w_p = 1e-9;
    }

    // Final de-homogenization
    double w_hom = solution_2.at<double>(3);
    if (std::abs(w_hom) > 1e-9) {
        double X = solution_2.at<double>(0) / w_hom;
        double Y = solution_2.at<double>(1) / w_hom;
        double Z = solution_2.at<double>(2) / w_hom;
        return (Z < 0) ? cv::Point3d(0, 0, 0) : cv::Point3d(X, Y, Z);
    }

    return cv::Point3d(0, 0, 0);
}

}


