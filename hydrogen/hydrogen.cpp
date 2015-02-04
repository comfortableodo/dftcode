# include <opencv2/core/core.hpp>
# include <eigen3/Eigen/Dense>
# include <eigen3/Eigen/Sparse>
# include <eigen3/unsupported/Eigen/KroneckerProduct>
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <vector>

# include "laplacian.hpp"

using namespace std;

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


//****************************************************************************80
// Prototypes for functions.
//****************************************************************************80

// Copied from eigen.tuxfamily.org to test discrete Laplacian.
void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n);
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename);

// Equivalent of Matlab linspace function.
std::vector<double> linspace(double min, double max, int n);
// Equivalent of Matlab meshgrid function.
static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, const cv::Mat &zgv,
              cv::Mat_<double> &X, cv::Mat_<double> &Y, cv::Mat_<double> &Z);
// Helper function that converts cv::Range into std::vector for integer values
// and calls meshgrid.
static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
                         const cv::Range &zgv, cv::Mat_<double> &X,
                         cv::Mat_<double> &Y, cv::Mat_<double> &Z);
// Convert a 2 dim. matrix to a vector.
std::vector<double> convertTwoDimMatrixToVector(cv::Mat_<double> matrix);


int main (int argc, char* argv[])
{

/*
 * ===========================================================================
 * Variable declarations.
 * ===========================================================================
 */

    int number_of_grid_points;
    int number_of_grid_points_cubed;
    double grid_spacing;
    int dimensions;

    std::vector<double> temp_vec;

    // Declarations of squares of vectors.
    std::vector<double> x_vec_squared;
    std::vector<double> y_vec_squared;
    std::vector<double> z_vec_squared;

    std::vector<double> r; // Distance from the centre.
    std::vector<double> v_ext; // External potential.

    dimensions = 3;
    number_of_grid_points = 10;
    number_of_grid_points_cubed = pow (number_of_grid_points, dimensions);

    // Not being used so far...
//    std::vector<int> one_vec (number_of_grid_points, 1);

/*
 * ===========================================================================
 * External potential.
 * ===========================================================================
 */

    std::vector<double> grid_points = linspace(-5, 5, number_of_grid_points);

    grid_spacing = grid_points[2]-grid_points[1];

    cv::Mat_<double> x, y, z;

/*
 * ===========================================================================
 * Debugging.
 * ===========================================================================
//    Print out the linspace grid_points.
    for (std::vector<double>::const_iterator i = grid_points.begin();
         i != grid_points.end(); ++i) std::cout << *i << ' ';
    Print the grid spacing
    std::cout << std::endl;
    std::cout << grid_spacing << std::endl;

//    Meshgrid test
    cv::Mat_<double> X, Y, Z;
//    The class cv::range is used to specify a row or a column span in a matrix
//    ( Mat ) and for many other purposes. Range(a,b) is basically the same as
//    a:b in  Matlab or a..b in Python.
//    meshgridTest(cv::Range(1,3), cv::Range(1,3), cv::Range(1,3), X, Y, Z);
*/

    //  Call Matlab's meshgrid function equivalent.
    meshgrid(cv::Mat(grid_points), cv::Mat(grid_points), cv::Mat(grid_points),
             x, y, z);

    //  Make X, Y, Z one dimensional. Or in other word, convert a matrix to a
    //  vector.
    std::vector<double> x_vec = convertTwoDimMatrixToVector(x);
    std::vector<double> y_vec = convertTwoDimMatrixToVector(y);
    std::vector<double> z_vec = convertTwoDimMatrixToVector(z);

    //  Square the x, y, z vectors.
    cv::multiply(x_vec, x_vec, x_vec_squared);
    cv::multiply(y_vec, y_vec, y_vec_squared);
    cv::multiply(y_vec, z_vec, z_vec_squared);

//    float test = cv::norm(Xvec);

    // Create the external potential.
    cv::add(x_vec_squared, y_vec_squared, temp_vec);
    cv::add(z_vec_squared, temp_vec, r);
    // Square root of r (seems complicated).
    std::transform(r.begin(), r.end(), r.begin(),
                    static_cast<double (*)(double)>(std::sqrt));
    // r=-1/v_ext
    cv::divide(-1, r, v_ext);

    //  And print them for debugging purposes.
//    for (std::vector<double>::const_iterator i = x_vec_squared.begin();
//         i != x_vec_squared.end(); ++i) std::cout << *i << ' ';
//    std::cout << std::endl;
//    for (std::vector<double>::const_iterator i = y_vec_squared.begin();
//         i != y_vec_squared.end(); ++i) std::cout << *i << ' ';
//    std::cout << std::endl;
//    for (std::vector<double>::const_iterator i = v_ext.begin();
//         i != v_ext.end(); ++i) std::cout << *i << ' ';
//    std::cout << std::endl;

/*
 * ===========================================================================
 * Kinetic energy.
 * ===========================================================================
 */

    double h = ( grid_points[2] - grid_points[1] );
    double *dense_laplacian_array;

    dense_laplacian_array = l1dd ( number_of_grid_points, h );
    r8mat_print ( number_of_grid_points, number_of_grid_points,
                  dense_laplacian_array, "  L1DD:" );


    Eigen::Map<Eigen::MatrixXd>
            dense_laplacian_matrix(dense_laplacian_array,number_of_grid_points,
               number_of_grid_points);

    std::cout << dense_laplacian_matrix << endl;

    Eigen::SparseMatrix<double> sparse_laplacian_matrix =
            dense_laplacian_matrix.sparseView();

    // Is this computationally demanding?
    sparse_laplacian_matrix = (-1)*sparse_laplacian_matrix;

//    Eigen::SparseMatrix<double> sparse = dense_laplacian_array.sparseView();
//    std::cout << sparse << std::endl;

    delete [] dense_laplacian_array;

    // Create the dense idenity matrix with dim. number_of_grid_points.
    Eigen::MatrixXd
            dense_identity_matrix(number_of_grid_points,number_of_grid_points);
    dense_identity_matrix =
            Eigen::MatrixXd::Identity(number_of_grid_points,number_of_grid_points);
    // Make dense identy matrix sparse.
    Eigen::SparseMatrix<double> sparse_identity_matrix =
            dense_identity_matrix.sparseView();

    std::cout << sparse_identity_matrix << std::endl;

    std::cout << sparse_laplacian_matrix << std::endl;

    // There must be a way to do this simpler...
    Eigen::SparseMatrix<double> sparse_laplacian_matrix_term1 =
            Eigen::kroneckerProduct(sparse_laplacian_matrix,sparse_identity_matrix);
    sparse_laplacian_matrix_term1 =
            kroneckerProduct(sparse_laplacian_matrix_3d,sparse_identity_matrix).eval();
    Eigen::SparseMatrix<double> sparse_laplacian_matrix_term2 =
            Eigen::kroneckerProduct(sparse_laplacian_matrix,sparse_identity_matrix);
    sparse_laplacian_matrix_term1 =
            kroneckerProduct(sparse_laplacian_matrix_3d,sparse_identity_matrix).eval();
    Eigen::SparseMatrix<double> sparse_laplacian_matrix_term3 =
            Eigen::kroneckerProduct(sparse_laplacian_matrix,sparse_identity_matrix);
    sparse_laplacian_matrix_term1 =
            kroneckerProduct(sparse_laplacian_matrix_3d,sparse_identity_matrix).eval();



    return 0;
}

// Equivalent of Matlab linspace function.
std::vector<double> linspace(double min, double max, int n)
{
    std::vector<double> result;
    // vector iterator
    int iterator = 0;

    for (int i = 0; i <= n-2; i++)
    {
        double temp = min + i*(max-min)/(floor((double)n) - 1);
        result.insert(result.begin() + iterator, temp);
        iterator += 1;
    }

    //iterator += 1;

    result.insert(result.begin() + iterator, max);
    return result;
}

// Equivalent of Matlab meshgrid function.
static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, const cv::Mat &zgv,
              cv::Mat_<double> &X, cv::Mat_<double> &Y, cv::Mat_<double> &Z)
{
    cv::repeat(xgv.reshape(1,1), ygv.total(), 1, X);
    cv::repeat(ygv.reshape(1,1).t(), 1, xgv.total(), Y);
    cv::repeat(zgv.reshape(1,1).t(), 1, xgv.total(), Z);
}

// Helper function that converts cv::Range into std::vector for integer values
// and calls meshgrid.
static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
                         const cv::Range &zgv, cv::Mat_<double> &X,
                         cv::Mat_<double> &Y, cv::Mat_<double> &Z)
{
    std::vector<double> t_x, t_y, t_z;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
    for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
    for (int i = zgv.start; i <= zgv.end; i++) t_z.push_back(i);

//    Debug
//    Print out t_x, t_y, t_z
    for (std::vector<double>::const_iterator i = t_x.begin();
         i != t_x.end(); ++i) std::cout << *i << ' ';

    std::cout << std::endl;

//    Call the meshgrid function
    meshgrid(cv::Mat_<double>(t_x), cv::Mat_<double>(t_y),
             cv::Mat_<double>(t_z), X, Y, Z);
}

// Convert a 2 dim. matrix to a vector.
std::vector<double> convertTwoDimMatrixToVector(cv::Mat_<double> matrix)
{
    double *p = matrix.ptr<double>(0); // Pointer to row 0 of X matrix
    // Construct a vector using a pointer.
    std::vector<double> vector(p, p+matrix.cols*matrix.rows); // Is this dangerous!?

    return vector;
}
