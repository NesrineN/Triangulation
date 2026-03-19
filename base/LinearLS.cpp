#include "LinearLS.h"
#include <complex>
#include <opencv2/opencv.hpp>

namespace Triangulation {

// function that performs the triangulation using the Linear Least Squares method:
// Here, we are setting X=(X,Y,Z,1) --> assuming point is not at infinity 
// aim: to reduce set of equations to 4 non-homogeneous equations with 3 unknowns only

cv::Point3d triangulate_Linear_LS(const cv::Vec3d& U, const cv::Vec3d& U_prime, const cv::Mat& P, const cv::Mat& P_prime){
    // extracting the coordinates from the u and u' vectors
    double u = U[0];
    double v = U[1];
    double u_p = U_prime[0];
    double v_p = U_prime[1];

    // creating the matrix A that is 4x4
    cv::Mat A = cv::Mat::zeros(4, 4, CV_64F);

    // filling it up
    // Row 0: uP3T-P1T
    A.row(0) = u*P.row(2) - P.row(0);
    // Row 1: vP3T-P2T
    A.row(1) = v*P.row(2) - P.row(1);
    // Row 2: u'P3'T - P1'T
    A.row(2) = u_p*P_prime.row(2) - P_prime.row(0);
    // Row 3: v'P3'T - P2'T
    A.row(3) = v_p*P_prime.row(2) - P_prime.row(1);

    // We separate into A' (first 3 columns) and b (negated 4th column)
    // A.colRange(start, end) is exclusive of the end index
    cv::Mat A_prime = A.colRange(0, 3); 
    cv::Mat b = -A.col(3);

    // We then Solve A'x = b using SVD pseudo-inverse method
    cv::Mat solution;
    bool success = cv::solve(A_prime, b, solution, cv::DECOMP_SVD);

    if (success) {
        double X = solution.at<double>(0);
        double Y = solution.at<double>(1);
        double Z = solution.at<double>(2);

        if (Z < 0) return cv::Point3d(0, 0, 0); // point is behind the camera

        return cv::Point3d(X, Y, Z);
    }
    
    else {
        return cv::Point3d(0, 0, 0);
    }
}

}





