#include "libOrsa/libNumerics/matrix.h"
#include "LinearEigen.h"
#include <iostream>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;


namespace Triangulation {

    Vec Triangulate_Poly(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr){

        // we compute F
        // we correct U and U_prime
        // std::pair<cv::Vec3d, cv::Vec3d> ComputeCorrectedPairs(const cv::Vec3d& U, const cv::Vec3d& U_prime, const cv::Mat& F)

        // we then use the corrected pair and P and P_prime to do triangulation using linear Eigen
        Vec result= Triangulation::Triangulate_Linear_Eigen(U_hat, U_hat_prime, P, P_prime);
        return result;
    }


}

