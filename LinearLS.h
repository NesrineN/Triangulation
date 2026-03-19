#ifndef LINEAR_LS_H
#define LINEAR_LS_H

#include "libOrsa/libNumerics/matrix.h"

// Using local typedefs to keep the interface clean
typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Triangulation {

    /**
     * @brief Solves a linear system Ax = b using the SVD Pseudo-inverse.
     * Useful for overdetermined systems (more equations than unknowns).
     */
    Vec solveSVD(const Mat& A, const Vec& b, double threshold = 1e-9);

    /**
     * @brief Performs 3D triangulation using the Linear Least Squares method.
     * Converts the homogeneous system AX=0 into a Euclidean system A'x = b.
     * * @param U Pixel coordinates (u, v) in the first image.
     * @param U_prime Pixel coordinates (u', v') in the second image.
     * @param P Projection matrix of the first camera.
     * @param P_prime Projection matrix of the second camera.
     * @return Vec The 3D coordinates (X, Y, Z). Returns (0,0,0) if point is behind camera.
     */
    Vec Triangulate_Linear_LS(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime);

} // namespace Triangulation

#endif // LINEAR_LS_H