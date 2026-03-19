#ifndef LINEAR_LS_H
#define LINEAR_LS_H

#include <opencv2/opencv.hpp>

namespace Triangulation {

    /**
     * @brief Performs 3D triangulation using the Linear Least Squares (Inhomogeneous) method.
     * * This method assumes the point is not at infinity (w=1) and solves the 
     * non-homogeneous system A'x = b using SVD decomposition.
     * * @param U Pixel coordinates in the first image (u, v, 1).
     * @param U_prime Pixel coordinates in the second image (u', v', 1).
     * @param P Projection matrix of the first camera (3x4).
     * @param P_prime Projection matrix of the second camera (3x4).
     * @return cv::Point3d The reconstructed 3D point, or (0,0,0) if reconstruction fails.
     */
    cv::Point3d triangulate_Linear_LS(
        const cv::Vec3d& U, 
        const cv::Vec3d& U_prime, 
        const cv::Mat& P, 
        const cv::Mat& P_prime
    );

} // namespace Triangulation

#endif // LINEAR_LS_H