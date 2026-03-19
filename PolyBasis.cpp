#include <gsl/gsl_poly.h>
#include "libOrsa/libNumerics/matrix.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <complex>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

// cv::decomposeProjectionMatrix(P0, K0, R0, T0); 

// function that takes the pairs u and u' and their fundamental matrix F and returns the corrected pairs u^ and u^'
std::pair<Vec, Vec> ComputeCorrectedPairs(const Vec& U, const Vec& U_prime, const Mat& F){
    
    // first we apply the rigid transformations to get F_transformed
    Mat L=TranslationMatrixToOrigin(U);
    Mat L_p=TranslationMatrixToOrigin(U_prime);

    Vec LeftEpipole=ComputeLeftEpipole(F);
    Vec RightEpipole=ComputeRightEpipole(F);

    Mat R=RotationMatrixToX(LeftEpipole);
    Mat R_p=RotationMatrixToX(RightEpipole);

    Mat F_transformed=TransformFundamentalMatrix(F, R, L, R_p, L_p);

    //F_transformed = 
    // ff'd -f'c -f'd
    // -fb    a    b
    // -fd    c    d 

    // then, we get the coefficients of F_transformed a,b,c,d,f,f_p
    double a=F_transformed(1,1);
    double b=F_transformed(1,2);
    double c=F_transformed(2,1);
    double d=F_transformed(2,2);
    double f=-(F_transformed(1,0)/b);
    double f_p=-(F_transformed(0,1)/c);

    // now, we find the best root that would minimize our cost function
    Vec roots=SolvePoly(a, b, c, d, f, f_p);
    double best_root=FindBestRoot(roots, a, b, c, d, f, f_p);

    // after finding t, we get the equations of the 2 epipolar lines lambda(t) and lambda(t')
    Vec lambda=ComputeLeftEpipolarLine(best_root, f);
    Vec lambda_p=ComputeRightEpipolarLine(best_root, a, b, c, d, f_p );

    // now, we find u^ and u^' which are the points on lambda and lambda_p closest to the origin
    Vec U_hat_old=FindClosestPointToOrigin(lambda);
    Vec U_hat_p_old=FindClosestPointToOrigin(lambda_p);

    // then we reverse the rigid transformations we applied at the beginning
    Vec U_hat=BackTransform(R, L, U_hat_old);
    Vec U_hat_p=BackTransform(R_p, L_p, U_hat_p_old);

    return std::pair<Vec, Vec>(U_hat, U_hat_p);
}

// the function above will use:

// function to do the rigid transformations --> TranslateToOrigin placing u=u'=(0,0,1) and place e1 and e2 on the x-axis such that:
// e1=(1,0,f) and e2=(1,0,f')

Mat TranslationMatrixToOrigin(const Vec& U){
    Mat result = Mat::eye(3, 3);
    result(0, 2) = -U(0);
	result(1, 2) = -U(1);
	return result;
} // returns translation matrix denoted by L in the paper . L*u gives the origin 

Mat RotationMatrixToX(const Vec& e)
{
    Mat result = Mat::eye(3, 3);
    
    // we calculate the magnitude of the 2D part of the epipole
    double norm = std::sqrt(e(0) * e(0) + e(1) * e(1));
    
    if (norm < 1e-9) {
        return result; // This means the epipole is already at the origin or invalid
    }

    double cos = e(0) / norm;
    double sin = e(1) / norm;

    // To satisfy sin(e1) + cos(e2) = 0:
    // We want to rotate the vector (ex, ey) to (norm, 0)
    result(0,0) = cos;
    result(0,1) = sin;
    result(1,0) = -sin;
    result(1,1) = cos;

    return result;
}

// function to compute the Fundamental Matrix F from the Projection Matrices P and P'
Mat ComputeFundamentalMatrix(const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr){
    Mat R=Rl*Rr.t();
    Vec T= Tl - (R*Tr);
    Mat Tx=Mat::zeros(3);
    Tx(0,1)=-T(2);
    Tx(0,2)=T(1);
    Tx(1,0)=T(2);
    Tx(1,2)=-T(0);
    Tx(2,0)=-T(1);
    Tx(2,1)=T(0);

    Mat E=Tx*R;
    Mat F=((K.inv()).t() * E)*(K.inv());

    return F;
}

// function to Compute the Fundamental Matrix after rigid transformations
// NOT SURE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Mat TransformFundamentalMatrix(const Mat& F, const Mat& R, const Mat& L, const Mat& R_p, const Mat L_p){
    Mat T = R*L;
    Mat T_p = R_p*L_p;

    Mat F_transformed= T_p*F*T.inv();

    return F_transformed;
}

// function to ComputeRightEpipole and ComputeLeftEpipole
// right epipole: we solve Fe=0 --> SVD
// left epipole: we solve e^T F=0 --> F^T e=0 --> SVD
Vec ComputeRightEpipole(const Mat& F){
    Mat W(3,3);
    Mat V(3,3);
    Vec S(3);
    F.SVD(W,S,V);

    int minIndex = 0;
    for (int i = 1; i < 3; ++i)
        if (S(i) < S(minIndex))
            minIndex = i;

    Vec e = V.col(minIndex);

    if (std::abs(e(2)) > 1e-9) {
        e(0) /= e(2);
        e(1) /= e(2);
        e(2) = 1.0;
    }

    return e;
}

cv::Vec3d ComputeLeftEpipole(const cv::Mat& F){
    Mat W(3,3);
    Mat V(3,3);
    Vec S(3);
    F.SVD(W,S,V);

    int minIndex = 0;
    for (int i = 1; i < 3; ++i)
        if (S(i) < S(minIndex))
            minIndex = i;

    Vec e = W.col(minIndex);

    if (std::abs(e(2)) > 1e-9) {
        e(0) /= e(2);
        e(1) /= e(2);
        e(2) = 1.0;
    }

    return e;
}

// function to solve the 6-degree polynomial 
std::vector<double> SolvePoly(const double a, const double b, const double c, const double d, const double f, const double f_p){
    double K = a*d - b*c;
    double f2 = f*f;
    double f4 = f2*f2;
    double f_prime2 = f_p * f_p;

    // Coefficients for L(t) = alpha*t^2 + beta*t + gamma
    double alpha = a*a + f_prime2*c*c;
    double beta  = 2*(a*b + f_prime2*c*d);
    double gamma = b*b + f_prime2*d*d;

    double c6 = -f4 * K * a_param * c;
    double c5 = alpha * alpha - f4 * K * (a_param*d + b*c);
    double c4 = 4 * (a_param*b + f_prime2*c*d) * alpha - f4 * K * b * d - 2 * f2 * K * a_param * c;
    double c3 = 2 * alpha * gamma + beta * beta - 2 * f2 * K * (a_param * d + b * c);
    double c2 = 2 * beta * gamma - 2 * f2 * K * b * d - K * a_param * c;
    double c1 = gamma * gamma - K * (a_param * d + b * c);
    double c0 = -K * b * d;

    double coeffs[7] = { c0, c1, c2, c3, c4, c5, c6 };

    std::vector<double> z(12);
    gsl_poly_complex_workspace * w = gsl_poly_complex_workspace_alloc (7);
    gsl_poly_complex_solve (coeffs, 7, w, z.data());
    gsl_poly_complex_workspace_free (w);

    return z;
}

// function that evaluates the equation s(t) given a t parameter
double EvaluateEquation(const double& t, const double a, const double b, const double c, const double d, const double f, const double f_p){
    double term1= (t*t) / (1+((t*f)*(t*f)));

    double term2_num= ((c*t)+d)*((c*t)+d);
    double term2_denom=((a*t)+b)*((a*t)+b) + (f_p * f_p)*term2_num;
    if (std::abs(term2_denom) < 1e-20) return term1;

    return (term1+(term2_num/term2_denom));
}

// function that decides which root gives the minimum s(t)
// we only want the real roots
double FindBestRoot(const Vec& roots, const double a, const double b, const double c, const double d, const double f, const double f_p){
    double minimum=1e99;
    double best_root;
    for(int i = 0; i < 6; i++) {
        double realPart = roots(2*i);
        double imagPart = roots(2*i + 1);

        double st=EvaluateEquation(realPart, a, b, c, d, f, f_p);
        if(st<minimum){
            minimum=st;
            best_root=realPart;
        }
    }

    return best_root;
}

// function that Constructs the epipolar lines lambda(t) and lambda'(t) from the root t that we found and the fundamental matrix transformed F'
// lambda(t)=(tf, 1, -t)
// lambda'(t)=(-f'(ct+d), (at+b), (ct+d))
Vec ComputeLeftEpipolarLine(const double& best_root, const double& f){
    return Vec((best_root*f), 1, (-best_root));
}

Vec ComputeRightEpipolarLine(const double& best_root, const double& a, const double& b, const double& c, const double& d, const double& f_p ){
    double x=((-f_p)*((c*best_root)+d));
    double y=(a*best_root)+b;
    double z=(c*best_root)+d;

    return Vec(x,y,z);
}

// function to Find the Point On the epipolar Line that is Closest To Origin u or u' --> that point is u^ or u^'
Vec FindClosestPointToOrigin(const Vec& lambda){
    // since the points U=U'=(0,0,1) are at the origin, the equation:
    // u^ = u - (normal)(distance from u to lambda) becomes very simple

    double distance=lambda(2)/((lambda(0)*lambda(0)) + (lambda(1)*lambda(1)));
    double x=-(lambda(0)*distance);
    double y=-(lambda(1)*distance);

    return Vec(x,y,1);
}

// function to reverse the rigid transformations on u^ and u^'
// FIX THIS!!!!!!!!!!!!!!! 
Vec BackTransform(const Mat& R, const Mat& L, const Vec& U_hat) {
    // Total forward transformation M = R * L
    Mat M = R * L;

    // Total backward transformation (Inverse)
    Mat M_inv = M.inverse();

    Vec u_hom(3);
    u_hom(0) = U_hat(0);
    u_hom(1) = U_hat(1);
    u_hom(2) = 1.0; 

    // Transforming back: Matrix * Vector = Vector
    Vec pixel_hom = M_inv * u_hom;

    // De-homogenizing
    double w = pixel_hom(2);
    
    // Safety check for points at infinity
    if(std::abs(w) < 1e-20) {
        return Vec(0.0, 0.0, 0.0); 
    }

    // Returning the coordinates in the original image plane
    Vec result(3);
    result(0) = pixel_hom(0) / w;
    result(1) = pixel_hom(1) / w;
    result(2) = 1.0;
    
    return result;
}