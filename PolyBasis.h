#ifndef POLY_BASIS_H
#define POLY_BASIS_H

#include <gsl/gsl_poly.h>
#include "libOrsa/libNumerics/matrix.h"
#include <iostream>
#include <complex>
#include <vector>

// Type Aliases for brevity
typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Poly
{
    /**
     * Core Function: Takes measured points u, u' and fundamental matrix F,
     * returns the "optimal" points u_hat and u_prime_hat that satisfy the
     * epipolar constraint perfectly while minimizing geometric distance.
     */
    std::pair<Vec, Vec> ComputeCorrectedPairs(const Vec& U, const Vec& U_prime, const Mat& F);

    // --- Geometric Transformation Helpers ---
    
    Mat TranslationMatrixToOrigin(const Vec& U);
    
    Mat RotationMatrixToX(const Vec& e);
    
    Mat ComputeFundamentalMatrix(const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr);
    
    Mat TransformFundamentalMatrix(const Mat& F, const Mat& R, const Mat& L, const Mat& R_p, const Mat& L_p);

    // --- Epipolar Geometry Helpers ---
    
    Vec ComputeRightEpipole(const Mat& F);
    
    Vec ComputeLeftEpipole(const Mat& F);

    // --- Polynomial & Optimization Logic ---
    
    std::vector<double> SolvePoly(double a, double b, double c, double d, double f, double f_p);
    
    double EvaluateEquation(const double& t, double a, double b, double c, double d, double f, double f_p);
    
    double FindBestRoot(const std::vector<double>& roots, double a, double b, double c, double d, double f, double f_p);

    // --- Line and Point Construction ---
    
    Vec ComputeLeftEpipolarLine(const double& best_root, const double& f);
    
    Vec ComputeRightEpipolarLine(const double& best_root, double a, double b, double c, double d, double f_p);
    
    Vec FindClosestPointToOrigin(const Vec& lambda);
    
    Vec BackTransform(const Mat& R, const Mat& L, const Vec& U_hat);

} // namespace Poly

#endif // POLY_BASIS_H