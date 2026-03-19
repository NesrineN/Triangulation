#include <iostream>
#include <tuple>

#include "LinearEigen.h"
#include "LinearLS.h"
#include "IterativeEigen.h"
#include "IterativeLS.h"
#include "Poly.h"

#include "libOrsa/libNumerics/matrix.h"
#include "CppUnitLite/TestHarness.h"


typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

// We set up synthetic stereo camera systems to test our methods:

// 1. We set 2 cameras that are aligned horizontally next to each other separated by a baseline distance : cam 0 is at the left and cam 1 is 1000 units away to the right from cam0
// this means no rotation and translation only in the second camera with respect to the first. C0 left camera center is at the world's origin (0,0,0)
// Function which returns the projection matrices P0 and P1 in the Horizontal Configuration case:

std::tuple<Mat, Mat, Mat, Mat, Mat, Vec, Vec> HorizontalConfiguration(){
    // P0 and P1 are originally identity matrices with an extra column 
	Mat P0L = Mat::eye(3); // Rl=Identity
	Mat P1L = Mat::eye(3); // Rr=Identity

    Mat Rl=P0L;
    Mat Rr=P1L;

    Vec P0R(0,0,0); // tl=0
    Vec Tl=P0R;
    Vec P1R(-1000,0,0); // tr=[-1000,0,0]
    Vec Tr=P1R;

    Mat P0=libNumerics::cat(P0L, P0R);
    Mat P1=libNumerics::cat(P1L, P1R);

    // Kl=Kr 
	Mat K = Mat::eye(3);
	K(0, 0) = 7291.67;
	K(1, 1) = 7291.67;
	K(0, 2) = 639.5;
	K(1, 2) = 511.5;

	P0 = K * P0;
	P1 = K * P1;
	return std::make_tuple(P0, P1, K, Rl, Rr, Tl, Tr);
}

// in here, we evaluate the error (distance) which is the norm of the result we obtained (X obtained) - expected result (X ground truth)
// ideally, the error should be zero but we allow a tolerance of 1% deviation away from the actual distance of the X ground truth
double CalculateError(const Vec& result, const Vec& expected_result)
{	
    return (result - expected_result).qnorm();
}

TEST(LinearEigenTest, HorizontalStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(2097.834, 511.500);
    Vec U_prime(639.500, 511.500);
    
    Vec result = Triangulation::Triangulate_Linear_Eigen(U,U_prime, P0, P1);

    Vec expected_result(1000.0, 0.0, 5000.0); // this is the coordinates of the 3D point X we are triangulating in the world coordinates 
	// distance should be = 0 with +- tolerance allowed
    // If the distance is smaller than the tolerance, the test Passes green. If the distance is larger, the test Fails red and tells us exactly how far off we were.
    
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    double error= CalculateError(result, expected_result);

    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(LinearLSTest, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(2097.834, 511.500);
    Vec U_prime(639.500, 511.500);
    
    Vec result = Triangulation::Triangulate_Linear_LS(U,U_prime, P0, P1);

    Vec expected_result(1000.0, 0.0, 5000.0); // this is the coordinates of the 3D point X we are triangulating in the world coordinates 
	// distance should be = 0 with +- tolerance allowed
    // If the distance is smaller than the tolerance, the test Passes green. If the distance is larger, the test Fails red and tells us exactly how far off we were.
    
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    double error= CalculateError(result, expected_result);

    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(IterativeEigenTest, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(2097.834, 511.500);
    Vec U_prime(639.500, 511.500);
    
    Vec result = Triangulation::Triangulate_Iterative_Eigen(U,U_prime, P0, P1);

    Vec expected_result(1000.0, 0.0, 5000.0); // this is the coordinates of the 3D point X we are triangulating in the world coordinates 
	// distance should be = 0 with +- tolerance allowed
    // If the distance is smaller than the tolerance, the test Passes green. If the distance is larger, the test Fails red and tells us exactly how far off we were.
    
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    double error= CalculateError(result, expected_result);

    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(IterativeLSTest, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();


    Vec U(2097.834, 511.500);
    Vec U_prime(639.500, 511.500);
    
    Vec result = Triangulation::Triangulate_Iterative_LS(U,U_prime, P0, P1);

    Vec expected_result(1000.0, 0.0, 5000.0); // this is the coordinates of the 3D point X we are triangulating in the world coordinates 
	// distance should be = 0 with +- tolerance allowed
    // If the distance is smaller than the tolerance, the test Passes green. If the distance is larger, the test Fails red and tells us exactly how far off we were.
    
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    double error= CalculateError(result, expected_result);

    DOUBLES_EQUAL(0.0, error, tolerance);
}

// In the horizontal setup, the epipoles are at infinity (since the cameras are parallel).
// The points u and u' have the exact same y-coord. the cost to minimize is already zero. 
// c6 coeff of the poly becomes 0 because the matrix F is missing the rank needed to generate a 6th-degree poly.
// sol: modify the degree of the poly from 6 to the actual degree depending on the non-zero coeff we ended up with 

TEST(PolyTest, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(2097.834, 511.500);
    Vec U_prime(639.500, 511.500);
    
    
    Vec result = Triangulation::Triangulate_Poly(U,U_prime, P0, P1, K, Rl, Rr, Tl, Tr);

    Vec expected_result(1000.0, 0.0, 5000.0); // this is the coordinates of the 3D point X we are triangulating in the world coordinates 
	// distance should be = 0 with +- tolerance allowed
    // If the distance is smaller than the tolerance, the test Passes green. If the distance is larger, the test Fails red and tells us exactly how far off we were.
    
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    double error= CalculateError(result, expected_result);

    DOUBLES_EQUAL(0.0, error, tolerance);
}

int main() { TestResult tr; return TestRegistry::runAllTests(tr); return 0;}