#include <gsl/gsl_poly.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <complex>

// function that takes the pairs u and u' and their fundamental matrix F and returns the corrected pairs u^ and u^'
std::pair<cv::Vec3d, cv::Vec3d> ComputeCorrectedPairs(const cv::Vec3d& U, const cv::Vec3d& U_prime, const cv::Mat& F){
    
    // first we apply the rigid transformations to get F_transformed
    cv::Mat L=TranslationMatrixToOrigin(U);
    cv::Mat L_p=TranslationMatrixToOrigin(U_prime);

    cv::Vec3d LeftEpipole=ComputeLeftEpipole(F);
    cv::Vec3d RightEpipole=ComputeRightEpipole(F);

    cv::Mat R=RotationMatrixToX(LeftEpipole);
    cv::Mat R_p=RotationMatrixToX(RightEpipole);

    cv::Mat F_transformed=TransformFundamentalMatrix(F, R, L, R_p, L_p);

    //F_transformed = 
    // ff'd -f'c -f'd
    // -fb    a    b
    // -fd    c    d 

    // then, we get the coefficients of F_transformed a,b,c,d,f,f_p
    double a=F_transformed.at<double>(1,1);
    double b=F_transformed.at<double>(1,2);
    double c=F_transformed.at<double>(2,1);
    double d=F_transformed.at<double>(2,2);
    double f=-(F_transformed.at<double>(1,0)/b);
    double f_p=-(F_transformed.at<double>(0,1)/c);

    // now, we find the best root that would minimize our cost function
    std::vector<double> roots=SolvePoly(a, b, c, d, f, f_p);
    double best_root=FindBestRoot(roots, a, b, c, d, f, f_p);

    // after finding t, we get the equations of the 2 epipolar lines lambda(t) and lambda(t')
    cv::Vec3d lambda=ComputeLeftEpipolarLine(best_root, f);
    cv::Vec3d lambda_p=ComputeRightEpipolarLine(best_root, a, b, c, d, f_p );

    // now, we find u^ and u^' which are the points on lambda and lambda_p closest to the origin
    cv::Vec3d U_hat_old=FindClosestPointToOrigin(lambda);
    cv::Vec3d U_hat_p_old=FindClosestPointToOrigin(lambda_p);

    // then we reverse the rigid transformations we applied at the beginning
    cv::Vec3d U_hat=BackTransform(R, L, U_hat_old);
    cv::Vec3d U_hat_p=BackTransform(R_p, L_p, U_hat_p_old);

    // at the end we use Linear LS to tringulate u^ and u^' 

    return std::pair<cv::Vec3d, cv::Vec3d>(U_hat, U_hat_p);
}

// the function above will use:

// function to do the rigid transformations --> TranslateToOrigin placing u=u'=(0,0,1) and place e1 and e2 on the x-axis such that:
// e1=(1,0,f) and e2=(1,0,f')

cv::Mat TranslationMatrixToOrigin(const cv::Vec3d& U){
    cv::Mat result = cv::Mat::eye(3, 3, CV_64F);
    result.at<double>(0, 2) = -p.x;
	result.at<double>(1, 2) = -p.y;
	return result;
} // returns translation matrix denoted by L in the paper 

cv::Mat RotationMatrixToX(const cv::Vec3d& e)
{
    cv::Mat result = cv::Mat::eye(3, 3, CV_64F);
    
    // we calculate the magnitude of the 2D part of the epipole
    double norm = std::sqrt(e.x * e.x + e.y * e.y);
    
    if (norm < 1e-9) {
        return result; // This means the epipole is already at the origin or invalid
    }

    double cos = e.x / norm;
    double sin = e.y / norm;

    // To satisfy sin(e1) + cos(e2) = 0:
    // We want to rotate the vector (ex, ey) to (norm, 0)
    result.at<double>(0,0) = cos;
    result.at<double>(0,1) = sin;
    result.at<double>(1,0) = -sin;
    result.at<double>(1,1) = cos;

    return result;
}

// function to Compute the Fundamental Matrix after rigid transformations
cv::Mat TransformFundamentalMatrix(const cv::Mat& F, const cv::Mat& R, const cv::Mat& L, const cv::Mat& R_p, const cv::Mat L_p){
    cv::Mat T = R*L;
    cv::Mat T_p = R_p*L_p;

    cv::Mat F_transformed=T_p*F*T.inv();

    return F_transformed;
}

// function to ComputeRightEpipole and ComputeLeftEpipole
// right epipole: we solve Fe=0 --> SVD
// left epipole: we solve e^T F=0 --> F^T e=0 --> SVD
cv::Vec3d ComputeRightEpipole(const cv::Mat& F){
    cv::Mat W, U_svd, VT;
    cv::SVD::compute(F, W, U_svd, VT); // The solution X is the last row of VT
    cv::Mat e = VT.row(VT.rows - 1); // solution is a 1x3 matrix

    return cv::Vec3d(e.at<double>(0), e.at<double>(1), e.at<double>(2));
}

cv::Vec3d ComputeLeftEpipole(const cv::Mat& F){
    cv::Mat W, U_svd, VT;
    cv::SVD::compute(F, W, U_svd, VT); // The solution X is the last row of VT
    cv::Mat e = U.col(U.cols - 1); // solution is a 1x3 matrix

    return cv::Vec3d(e.at<double>(0), e.at<double>(1), e.at<double>(2));
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
double FindBestRoot(const std::vector<double>& roots, const double a, const double b, const double c, const double d, const double f, const double f_p){
    double minimum=1e99;
    double best_root;
    for(int i = 0; i < 6; i++) {
        double realPart = roots[2*i];
        double imagPart = roots[2*i + 1];

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
cv::Vec3d ComputeLeftEpipolarLine(const double& best_root, const double& f){
    return cv::Vec3d((best_root*f), 1, (-best_root));
}

cv::Vec3d ComputeRightEpipolarLine(const double& best_root, const double& a, const double& b, const double& c, const double& d, const double& f_p ){
    double x=((-f_p)*((c*best_root)+d));
    double y=(a*best_root)+b;
    double z=(c*best_root)+d;

    return cv::Vec3d(x,y,z);
}

// function to Find the Point On the epipolar Line that is Closest To Origin u or u' --> that point is u^ or u^'
cv::Vec3d FindClosestPointToOrigin(const cv::Vec3d& lambda){
    // since the points U=U'=(0,0,1) are at the origin, the equation:
    // u^ = u - (normal)(distance from u to lambda) becomes very simple

    double distance=lambda.z/((lambda.x*lambda.x) + (lambda.y*lambda.y));
    double x=-(lambda.x*distance);
    double y=-(lambda.y*distance);

    return cv::Vec3d(x,y,1);
}

// function to reverse the rigid transformations on u^ and u^'
cv::Vec3d BackTransform(const cv::Mat& R, const cv::Mat& L, const cv::Vec3d& U_hat) {
    // The total forward transformation matrix
    cv::Mat M = R * L;

    // The total backward transformation matrix
    cv::Mat M_inv = M.inv();

    // To get the final pixel:
    cv::Mat pixel_hom = M_inv * (cv::Mat_<double>(3,1) << U_hat[0], U_hat[1], 1.0); // 3x1 

    double w = pixel_hom.at<double>(2);
    if(std::abs(w)< 1e-20){return cv::Vec3d(0,0,0);}
    
    return cv::Vec3d(pixel_hom.at<double>(0)/w, pixel_hom.at<double>(1)/w, 1);
}