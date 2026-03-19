#ifndef ITERATIVE_EIGEN_H
#define ITERATIVE_EIGEN_H

#include <opencv2/opencv.hpp>

namespace Triangulation {

    /**
     * @brief Performs 3D triangulation using the Iterative Eigen Method.
     * * @param U Pixel coordinates in the first image (u, v, 1).
     * @param U_prime Pixel coordinates in the second image (u', v', 1).
     * @param P Projection matrix of the first camera (3x4).
     * @param P_prime Projection matrix of the second camera (3x4).
     * @return cv::Point3d The reconstructed 3D point, or (0,0,0) if failed/behind camera.
     */
    cv::Point3d triangulate_Iterative_Eigen(
        const cv::Vec3d& U, 
        const cv::Vec3d& U_prime, 
        const cv::Mat& P, 
        const cv::Mat& P_prime
    );

} // namespace Triangulation

#endif // ITERATIVE_EIGEN_H