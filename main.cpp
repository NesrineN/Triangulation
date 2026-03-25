#include <iostream>
#include <tuple>
#include <chrono>
#include <functional>
#include <random>
#include <iomanip>

#include "LinearEigen.h"
#include "LinearLS.h"
#include "IterativeEigen.h"
#include "IterativeLS.h"
#include "Poly.h"
#include "PolyAbs.h"
#include "Kanatani.h"
#include "HigherOrder.h"

#include "libOrsa/libNumerics/matrix.h"
#include "CppUnitLite/TestHarness.h"


typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

// create here a vector array where we store all the results of the errors for each method and each configuration :
// Vec(0): Linear Eigen, Vec(1): Linear LS, Vec(2): Iterative Eigen, Vec(3): Iterative LS, Vec(4): Poly , Vec(5): Poly-Abs, Vec(6): Kanatani, Vec(7): Higher order correction
Vec Errors_Horizontal_noiseless(8);
Vec Errors_C2Left_noiseless(8);

Vec Errors_Horizontal_noisy(8);
Vec Errors_C2Left_noisy(8);



// in here, we evaluate the error (distance) which is the norm of the result we obtained (X obtained) - expected result (X ground truth)
// ideally, the error should be zero but we allow a tolerance of 1% deviation away from the actual distance of the X ground truth
double CalculateError(const Vec& result, const Vec& expected_result)
{	
    return (std::sqrt((result - expected_result).qnorm()));
}

void AddGaussianNoise(Vec& U, double sigma) {
    if(sigma==0){
        return;
    }
    static std::mt19937 gen(42); // seed for reproducible tests
    std::normal_distribution<double> dist(0.0, sigma);

    U(0) += dist(gen); // Adding noise to x
    U(1) += dist(gen); // Adding noise to y
}

template <typename Func, typename... Args>

// A generic function to run and validate any triangulation method
double RunTriangulationTest(
    const std::string& testName,
    Func triangulateFunc, 
    const Vec& expected_result,
    Args&&... args  
) {
    // 1. Timing
    auto start = std::chrono::high_resolution_clock::now();
    Vec result = triangulateFunc(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 2. Error Calculation
    double error = CalculateError(result, expected_result);
    double percentage_error = error / std::sqrt(expected_result.qnorm());
    
    // 3. Reporting
    std::cout << "\n--- " << testName << " ---" << std::endl;
    std::cout << "Execution Time: " << duration.count() << " us" << std::endl;
    std::cout << "Reconstruction Error: " << error << std::endl;
    std::cout << "Relative Error: " << (percentage_error * 100.0) << "%" << std::endl;

    // // 4. Assertion (using a 0.1% tolerance as in your code)
    // double max_percentage_error = 0.001;
    // double tolerance = expected_result.qnorm() * max_percentage_error;
    // DOUBLES_EQUAL(0.0, error, tolerance);
    return error;
}

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

std::tuple<Mat, Mat, Mat, Mat, Mat, Vec, Vec> C2RotatedLeft(){

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

	P0(0, 0) = 0.999701;
	P0(0, 1) = 0.0174497;
	P0(0, 2) = -0.017145;
	P0(1, 0) = -0.0171452;
	P0(1, 1) = 0.999695;
	P0(1, 2) = 0.0177517;
	P0(2, 0) = 0.0174497;
	P0(2, 1) = -0.0174524;
	P0(2, 2) = 0.999695;

	Mat K = Mat::eye(3);
	K(0, 0) = 7291.67;
	K(1, 1) = 7291.67;
	K(0, 2) = 639.5;
	K(1, 2) = 511.5;

	P0 = K * P0;
	P1 = K * P1;
	return std::make_tuple(P0, P1, K, Rl, Rr, Tl, Tr);

}

//--------------------------------------------------------------------------------------------------------------------------------

TEST(LinearEigenTest, HorizontalStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); // this is the coordinates of the 3D point X we are triangulating in the world coordinates 
	// distance should be = 0 with +- tolerance allowed
    // If the distance is smaller than the tolerance, the test Passes green. If the distance is larger, the test Fails red and tells us exactly how far off we were.

    double error=RunTriangulationTest("Linear Eigen - Horizontal Stereo", Triangulation::Triangulate_Linear_Eigen, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_Horizontal_noiseless(0)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(LinearLSTest, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 
    
    double error=RunTriangulationTest("Linear LS - Horizontal Stereo", Triangulation::Triangulate_Linear_LS, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_Horizontal_noiseless(1)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(IterativeEigenTest, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);  

    double error=RunTriangulationTest("Iterative Eigen - Horizontal Stereo", Triangulation::Triangulate_Iterative_Eigen, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_Horizontal_noiseless(2)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(IterativeLSTest, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();


    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 

    double error=RunTriangulationTest("Iterative LS - Horizontal Stereo", Triangulation::Triangulate_Iterative_LS, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_Horizontal_noiseless(3)= error / expected_result.qnorm();
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

    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 

    double error=RunTriangulationTest("Poly - Horizontal Stereo", Triangulation::Triangulate_Poly, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_Horizontal_noiseless(4)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(PolyAbsTest, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 

    double error=RunTriangulationTest("Poly-Abs - Horizontal Stereo", Triangulation::Triangulate_Poly_Abs, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_Horizontal_noiseless(5)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(Kanatani, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);
    
    double error=RunTriangulationTest("Kanatani - Horizontal Stereo", Triangulation::Triangulate_Kanatani, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_Horizontal_noiseless(6)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(HigherOrder, HorizontalStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);
    
    double error=RunTriangulationTest("Higher Order Optimal Correction - Horizontal Stereo", Triangulation::Triangulate_HigherOrder, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_Horizontal_noiseless(7)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

// ---------------------------------------------------------------------------------------------------------------------

TEST(LinearEigenTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 

    double error=RunTriangulationTest("Linear Eigen - C2 Rotated Left Stereo", Triangulation::Triangulate_Linear_Eigen, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noiseless(0)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(LinearLSTest, C2RotatedLeftStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 
    
    double error=RunTriangulationTest("Linear LS - C2 Rotated Left Stereo", Triangulation::Triangulate_Linear_LS, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noiseless(1)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(IterativeEigenTest, C2RotatedLeftStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 
    
    double error=RunTriangulationTest("Iterative Eigen - C2 Rotated Left Stereo", Triangulation::Triangulate_Iterative_Eigen, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noiseless(2)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(IterativeLSTest, C2RotatedLeftStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();


    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 
    
    double error=RunTriangulationTest("Iterative LS - C2 Rotated Left Stereo", Triangulation::Triangulate_Iterative_LS, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noiseless(3)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(PolyTest, C2RotatedLeftStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 
    
    double error=RunTriangulationTest("Poly - C2 Rotated Left Stereo", Triangulation::Triangulate_Poly, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noiseless(4)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(PolyAbsTest, C2RotatedLeftStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 
    
    double error=RunTriangulationTest("Poly-Abs - C2 Rotated Left Stereo", Triangulation::Triangulate_Poly_Abs, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noiseless(5)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(Kanatani, C2RotatedLeftStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 
    
    double error=RunTriangulationTest("Kanatani - C2 Rotated Left Stereo", Triangulation::Triangulate_Kanatani, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noiseless(6)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(HigherOrder, C2RotatedLeftStereo)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 
    
    double error=RunTriangulationTest("Higher Order Optimal Correction - C2 Rotated Left Stereo", Triangulation::Triangulate_HigherOrder, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001;
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noiseless(7)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

// ---------------------------------------------------------------------------------------------------------------------

TEST(NoisyLinearEigenTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 

    // adding noise:
    double sigma = 1.5; // 1.5 pixel noise
    AddGaussianNoise(U, sigma);
    AddGaussianNoise(U_prime, sigma);

    double error=RunTriangulationTest("Noisy Linear Eigen - C2 Rotated Left Stereo", Triangulation::Triangulate_Linear_Eigen, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001; // might need to increase the tolerance with noise 
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noisy(0)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(NoisyLinearLSTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);

    // adding noise:
    double sigma = 1.5; // 1.5 pixel noise
    AddGaussianNoise(U, sigma);
    AddGaussianNoise(U_prime, sigma);

    double error=RunTriangulationTest("Noisy Linear LS - C2 Rotated Left Stereo", Triangulation::Triangulate_Linear_LS, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001; // might need to increase the tolerance with noise 
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noisy(1)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(NoisyIterativeEigenTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 

    // adding noise:
    double sigma = 1.5; // 1.5 pixel noise
    AddGaussianNoise(U, sigma);
    AddGaussianNoise(U_prime, sigma);

    double error=RunTriangulationTest("Noisy Iterative Eigen - C2 Rotated Left Stereo", Triangulation::Triangulate_Iterative_Eigen, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001; // might need to increase the tolerance with noise 
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noisy(2)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(NoisyIterativeLSTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);

    // adding noise:
    double sigma = 1.5; // 1.5 pixel noise
    AddGaussianNoise(U, sigma);
    AddGaussianNoise(U_prime, sigma);

    double error=RunTriangulationTest("Noisy Iterative LS - C2 Rotated Left Stereo", Triangulation::Triangulate_Iterative_LS, expected_result, U, U_prime, P0, P1);
    double max_percentage_error = 0.001; // might need to increase the tolerance with noise 
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noisy(3)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(NoisyPolyTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);

    // adding noise:
    double sigma = 1.5; // 1.5 pixel noise
    AddGaussianNoise(U, sigma);
    AddGaussianNoise(U_prime, sigma);

    double error=RunTriangulationTest("Noisy Poly - C2 Rotated Left Stereo", Triangulation::Triangulate_Poly, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001; // might need to increase the tolerance with noise 
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noisy(4)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(NoisyPolyAbsTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);

    // adding noise:
    double sigma = 1.5; // 1.5 pixel noise
    AddGaussianNoise(U, sigma);
    AddGaussianNoise(U_prime, sigma);

    double error=RunTriangulationTest("Noisy Poly-Abs - C2 Rotated Left Stereo", Triangulation::Triangulate_Poly_Abs, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001; // might need to increase the tolerance with noise 
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noisy(5)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(NoisyKanataniTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0); 

    // adding noise:
    double sigma = 1.5; // 1.5 pixel noise
    AddGaussianNoise(U, sigma);
    AddGaussianNoise(U_prime, sigma);

    double error=RunTriangulationTest("Noisy Kanatani - C2 Rotated Left Stereo", Triangulation::Triangulate_Kanatani, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001; // might need to increase the tolerance with noise 
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noisy(6)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

TEST(NoisyHigherOrderTest, C2RotatedLeftStereo)
{   Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = C2RotatedLeft();

    Vec U(878.821, 634.619);
    Vec U_prime(274.917, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);

    // adding noise:
    double sigma = 1.5; // 1.5 pixel noise
    AddGaussianNoise(U, sigma);
    AddGaussianNoise(U_prime, sigma);

    double error=RunTriangulationTest("Noisy Higher Order - C2 Rotated Left Stereo", Triangulation::Triangulate_HigherOrder, expected_result, U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
    double max_percentage_error = 0.001; // might need to increase the tolerance with noise 
    double tolerance = expected_result.qnorm() * max_percentage_error;
    Errors_C2Left_noisy(7)= error / expected_result.qnorm();
    DOUBLES_EQUAL(0.0, error, tolerance);
}

// ----------------------------------------------------------------------------------------------------------------------------------------

// testing the methods for different noise levels and adding the errors to a vector for the horizontal stereo:
const int num_sigma = 11;
const int num_methods = 8;
Vec Err(num_sigma * num_methods);

TEST(TriangulationBatch, NoiseRobustness)
{
    Mat P0, P1, K, Rl, Rr;
    Vec Tl(3);
    Vec Tr(3);
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = HorizontalConfiguration();

    Vec U(1004.0835, 511.5);
    Vec U_prime(274.9165, 511.5);
    Vec expected_result(500.0, 0.0, 10000.0);
    Vec result(3);
    double error;
    double percentage_error;

    for(int s = 0; s <= 10; ++s) {
        double sigma=s;

        int i = s * 8;

        double sum_lineigen = 0, sum_linls= 0, sum_iteigen=0, sum_itls=0, sum_poly=0, sum_polyabs=0, sum_kanatani=0, sum_higherorder=0;
        int trials = 100;

        for(int t = 0; t < trials; ++t) {
            // 1. Resetting U, U_prime
            // 2. Adding Noise
            // 3. Running Triangulation methods

            U(0)=1004.0835;
            U(1)=511.5;
            U_prime(0)=274.9165;
            U_prime(1)= 511.5;

            AddGaussianNoise(U, sigma);
            AddGaussianNoise(U_prime, sigma);

            // 2. Running Triangulations
            result=Triangulation::Triangulate_Linear_Eigen(U, U_prime, P0, P1);
            error = CalculateError(result, expected_result);
            percentage_error = (error / std::sqrt(expected_result.qnorm())) *100;
            // Err(i)=percentage_error;
            sum_lineigen += percentage_error;

            result=Triangulation::Triangulate_Linear_LS(U, U_prime, P0, P1);
            error = CalculateError(result, expected_result);
            percentage_error = (error / std::sqrt(expected_result.qnorm())) *100;
            // Err(i+1)=percentage_error;
            sum_linls += percentage_error;

            result=Triangulation::Triangulate_Iterative_Eigen(U, U_prime, P0, P1);
            error = CalculateError(result, expected_result);
            percentage_error = (error / std::sqrt(expected_result.qnorm())) *100;
            // Err(i+2)=percentage_error;
            sum_iteigen += percentage_error;

            result=Triangulation::Triangulate_Iterative_LS(U, U_prime, P0, P1);
            error = CalculateError(result, expected_result);
            percentage_error = (error / std::sqrt(expected_result.qnorm())) *100;
            // Err(i+3)=percentage_error;
            sum_itls += percentage_error;

            result=Triangulation::Triangulate_Poly(U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
            error = CalculateError(result, expected_result);
            percentage_error = (error / std::sqrt(expected_result.qnorm())) *100;
            // Err(i+4)=percentage_error;
            sum_poly += percentage_error;
            
            result=Triangulation::Triangulate_Poly_Abs(U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
            error = CalculateError(result, expected_result);
            percentage_error = (error / std::sqrt(expected_result.qnorm())) *100;
            // Err(i+5)=percentage_error;
            sum_polyabs += percentage_error;

            result=Triangulation::Triangulate_Kanatani(U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
            error = CalculateError(result, expected_result);
            percentage_error = (error / std::sqrt(expected_result.qnorm())) *100;
            // Err(i+6)=percentage_error;
            sum_kanatani += percentage_error;

            result=Triangulation::Triangulate_HigherOrder(U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
            error = CalculateError(result, expected_result);
            percentage_error = (error / std::sqrt(expected_result.qnorm())) *100;
            // Err(i+7)=percentage_error;
            sum_higherorder += percentage_error;
        }

        Err(i)=sum_lineigen / trials;
        Err(i+1)=sum_linls / trials;
        Err(i+2)=sum_iteigen / trials;
        Err(i+3)=sum_itls / trials;
        Err(i+4)=sum_poly / trials;
        Err(i+5)=sum_polyabs / trials;
        Err(i+6)=sum_kanatani / trials;
        Err(i+7)=sum_higherorder / trials;
    }

    std::cout << "\n" << std::string(140, '=') << "\n";
    std::cout << std::left << std::setw(8)  << "Sigma" 
            << std::setw(16) << "Lin_Eigen" 
            << std::setw(16) << "Lin_LS" 
            << std::setw(16) << "Iter_Eigen" 
            << std::setw(16) << "Iter_LS" 
            << std::setw(16) << "Poly"
            << std::setw(16) << "Poly_Abs" 
            << std::setw(16) << "Kanatani" 
            << std::setw(16) << "HigherOrd" << "\n";
    std::cout << std::string(140, '-') << "\n";

    // Looping through the 11 stored noise levels
    for (int s = 0; s <= 10; ++s) {
        int b = s * 8; 
        std::cout << std::left << std::setw(8) << s 
                << std::fixed << std::setprecision(6) 
                << std::setw(16) << Err(b)
                << std::setw(16) << Err(b + 1)
                << std::setw(16) << Err(b + 2)
                << std::setw(16) << Err(b + 3)
                << std::setw(16) << Err(b + 4)
                << std::setw(16) << Err(b + 5)
                << std::setw(16) << Err(b + 6)
                << std::setw(16) << Err(b + 7) << "\n";
    }
    std::cout << std::string(140, '=') << "\n";
}

int main() { TestResult tr; return TestRegistry::runAllTests(tr); return 0;}