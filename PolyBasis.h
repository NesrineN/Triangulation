#ifndef POLY_BASIS_H
#define POLY_BASIS_H

#include "libOrsa/libNumerics/matrix.h"
#include <vector>
#include <utility>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Poly {

    Mat TranslationMatrixToOrigin(const Vec& U);
    Mat RotationMatrixToX(const Vec& e);

    Mat ComputeFundamentalMatrix(const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr);
    Mat TransformFundamentalMatrix(const Mat& F, const Mat& R, const Mat& L, const Mat& R_p, const Mat L_p);
    Vec ComputeRightEpipole(const Mat& F);
    Vec ComputeLeftEpipole(const Mat& F);

    std::vector<double> SolvePoly(const double a, const double b, const double c, const double d, const double f, const double f_p);
    std::vector<double> SolvePolyAbs(const double a, const double b, const double c, const double d, const double f, const double f_p);

    double EvaluateEquation(const double& t, const double a, const double b, const double c, const double d, const double f, const double f_p);
    double EvaluateEquationAbs(const double& t, const double a, const double b, const double c, const double d, const double f, const double f_p);
    double FindBestRoot(const std::vector<double>& roots, const double a, const double b, const double c, const double d, const double f, const double f_p);
    double FindBestRootAbs(const std::vector<double>& roots, const double a, const double b, const double c, const double d, const double f, const double f_p);

    Vec ComputeLeftEpipolarLine(const double& best_root, const double& f);
    Vec ComputeRightEpipolarLine(const double& best_root, const double& a, const double& b, const double& c, const double& d, const double& f_p);
    Vec FindClosestPointToOrigin(const Vec& lambda);
    Vec BackTransform(const Mat& R, const Mat& L, const Vec& U_hat);

    std::pair<Vec, Vec> ComputeCorrectedPairs(const Vec& U, const Vec& U_prime, const Mat& F);
    std::pair<Vec, Vec> ComputeCorrectedPairsAbs(const Vec& U, const Vec& U_prime, const Mat& F);

} // namespace Poly

#endif // POLY_BASIS_H