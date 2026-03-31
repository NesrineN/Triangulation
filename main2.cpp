#include <iostream>
#include <tuple>
#include <chrono>
#include <functional>
#include <random>
#include <iomanip>
#include <cmath>


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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::tuple<Mat, Mat, Mat, Mat, Mat, Vec, Vec> BothRotatedConfiguration() {
    double angle = 2.0 * M_PI / 180.0; // 2 degrees
    double c = cos(angle);
    double s = sin(angle);

    // Rotation Matrix for Camera 1 (Rotate Right around Y)
    Mat Rl = Mat::eye(3);
    Rl(0, 0) = c;  Rl(0, 2) = s;
    Rl(2, 0) = -s; Rl(2, 2) = c;

    // Rotation Matrix for Camera 2 (Rotate Left around Y)
    // To rotate left, we use -angle. cos(-a) = cos(a), sin(-a) = -sin(a)
    Mat Rr = Mat::eye(3);
    Rr(0, 0) = c;   Rr(0, 2) = -s;
    Rr(2, 0) = s;  Rr(2, 2) = c;

    // Translations
    Vec Tl(0, 0, 0);       // C1 at Origin
    Vec Tr(-1000, 0, 0);   // C2 moved 1000 units left

    // Build Projection Matrices [R | t]
    Mat P0 = libNumerics::cat(Rl, Tl);
    Mat P1 = libNumerics::cat(Rr, Tr);

    // Intrinsic Matrix K
    Mat K = Mat::eye(3);
    K(0, 0) = 7291.67; K(1, 1) = 7291.67;
    K(0, 2) = 639.5;   K(1, 2) = 511.5;

    // Apply K to get the final P matrices
    P0 = K * P0;
    P1 = K * P1;

    return std::make_tuple(P0, P1, K, Rl, Rr, Tl, Tr);
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
    std::tie(P0, P1, K, Rl, Rr, Tl, Tr) = BothRotatedConfiguration();

    // 1. Define the 3D point in Homogeneous coordinates (add a 1.0 at the end)
    Vec X_world(4);
    X_world(0)=500.0;
    X_world(1)=0.0;
    X_world(2)=10000.0;
    X_world(3)= 1.0; 

    // 2. Project onto Camera 0 (Left)
    Vec u_homog = P0 * X_world;
    double u = u_homog(0) / u_homog(2);
    double v = u_homog(1) / u_homog(2);
    Vec U(u, v);

    // 3. Project onto Camera 1 (Right)
    Vec up_homog = P1 * X_world;
    double u_prime = up_homog(0) / up_homog(2);
    double v_prime = up_homog(1) / up_homog(2);
    Vec U_prime(u_prime, v_prime);
    Vec expected_result(500.0, 0.0, 10000.0); 
    double norm_expected = std::sqrt(expected_result.qnorm());

    Vec result(3);
    double error;

    for(int s = 0; s <= 10; ++s) {
        double sigma=s;

        int i = s * 8;

        double sum_lineigen = 0, sum_linls= 0, sum_iteigen=0, sum_itls=0, sum_poly=0, sum_polyabs=0, sum_kanatani=0, sum_higherorder=0;
        int trials = 100;

        for(int t = 0; t < trials; ++t) {
            // 1. Resetting U, U_prime
            // 2. Adding Noise
            // 3. Running Triangulation methods

            U(0)=u;
            U(1)=v;
            U_prime(0)= u_prime;
            U_prime(1)= v_prime;

            AddGaussianNoise(U, sigma);
            AddGaussianNoise(U_prime, sigma);

            // 2. Running Triangulations
            result=Triangulation::Triangulate_Linear_Eigen(U, U_prime, P0, P1);
            error = CalculateError(result, expected_result);
            sum_lineigen += error;

            result=Triangulation::Triangulate_Linear_LS(U, U_prime, P0, P1);
            error = CalculateError(result, expected_result);
            sum_linls += error;

            result=Triangulation::Triangulate_Iterative_Eigen(U, U_prime, P0, P1);
            error = CalculateError(result, expected_result);
            sum_iteigen += error;

            result=Triangulation::Triangulate_Iterative_LS(U, U_prime, P0, P1);
            error = CalculateError(result, expected_result);
            sum_itls += error;

            result=Triangulation::Triangulate_Poly(U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
            error = CalculateError(result, expected_result);
            sum_poly += error;
            
            result=Triangulation::Triangulate_Poly_Abs(U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
            error = CalculateError(result, expected_result);
            sum_polyabs += error;

            result=Triangulation::Triangulate_Kanatani(U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
            error = CalculateError(result, expected_result);
            sum_kanatani += error;

            result=Triangulation::Triangulate_HigherOrder(U, U_prime, P0, P1, K, Rl, Rr, Tl, Tr);
            error = CalculateError(result, expected_result);
            sum_higherorder += error;
        }

        Err(i)=((sum_lineigen / trials)/norm_expected)*100;
        Err(i+1)=((sum_linls / trials)/norm_expected)*100;
        Err(i+2)=((sum_iteigen / trials)/norm_expected)*100;
        Err(i+3)=((sum_itls / trials)/norm_expected)*100;
        Err(i+4)=((sum_poly / trials)/norm_expected)*100;
        Err(i+5)=((sum_polyabs / trials)/norm_expected)*100;
        Err(i+6)=((sum_kanatani / trials)/norm_expected)*100;
        Err(i+7)=((sum_higherorder / trials)/norm_expected)*100;
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

int main() { TestResult tr; return TestRegistry::runAllTests(tr);}