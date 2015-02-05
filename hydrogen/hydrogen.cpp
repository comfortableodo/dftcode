# include <eigen3/Eigen/Dense>
# include <eigen3/Eigen/Sparse>
# include <eigen3/Eigen/StdVector>
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

//// Convert a 2 dim. matrix to a vector.
//Eigen::VectorXd convertTwoDimMatrixToVector(Eigen::MatrixXd matrix);

int main (int argc, char* argv[])
{

//****************************************************************************80
// Variable declarations.
//****************************************************************************80

    // Variables defining the grid and the box.
    int number_of_grid_points;
    int number_of_grid_points_cubed;
    double grid_spacing;
    int dimensions;
    int box_side_length;

    // Values for calculation.
    // Should be put into an external file and read by the program at
    // later stage.
    dimensions = 3;
    box_side_length = 5;
    number_of_grid_points = 3;
    number_of_grid_points_cubed = pow(number_of_grid_points, dimensions);

    // Declarations of vectors.
    Eigen::VectorXd r(number_of_grid_points_cubed); // Distance from the centre.

//    Eigen::VectorXd v_ext; // External potential.
    Eigen::VectorXd v_ext(number_of_grid_points_cubed);


//****************************************************************************80
// External potential.
//****************************************************************************80


    // Vector containing the grid points.
    Eigen::VectorXd grid_points;
    grid_points.setLinSpaced(number_of_grid_points,
                             (-1)*box_side_length,
                             box_side_length);

    // Grid spacing:
    grid_spacing = grid_points[2]-grid_points[1];
    // Print out the linspace grid_points.
//    std::cout << grid_points << std::endl;

    //  Call Matlab's meshgrid function equivalent.
    Eigen::MatrixXd x_matrix;
    Eigen::MatrixXd y_matrix;
    Eigen::MatrixXd z_matrix;

    x_matrix = Eigen::RowVectorXd::LinSpaced(number_of_grid_points,
                                       (-1)*box_side_length,
                                       box_side_length).
            replicate(number_of_grid_points,1);
    y_matrix = Eigen::RowVectorXd::LinSpaced(number_of_grid_points,
                                       (-1)*box_side_length,
                                       box_side_length).
            replicate(number_of_grid_points,1);
    z_matrix = Eigen::RowVectorXd::LinSpaced(number_of_grid_points,
                                       (-1)*box_side_length,
                                       box_side_length).
            replicate(number_of_grid_points,1);


    Eigen::Map<Eigen::VectorXd> x_vec(x_matrix.data(),x_matrix.size());
    Eigen::Map<Eigen::VectorXd> y_vec(y_matrix.data(),y_matrix.size());
    Eigen::Map<Eigen::VectorXd> z_vec(z_matrix.data(),z_matrix.size());

    std::cout << "x vector " << std::endl;
    std::cout << x_vec << std::endl;

    std::cout << "y vector " << std::endl;
    std::cout << y_vec << std::endl;

    std::cout << "z vector " << std::endl;
    std::cout << z_vec << std::endl;

    Eigen::VectorXd x_vec_squared;
    Eigen::VectorXd y_vec_squared;
    Eigen::VectorXd z_vec_squared;

    //  Square the x, y, z vectors.
    x_vec_squared = x_vec.array().square();
    y_vec_squared = y_vec.array().square();
    z_vec_squared = z_vec.array().square();

    std::cout << "x vector squared:" << std::endl;
    std::cout << x_vec_squared << std::endl;

    // Create the external potential.
    r = x_vec_squared + y_vec_squared + z_vec_squared;

    // sqrt(r)
    r = r.array().sqrt();

    // r=-1/v_ext
    v_ext = (-0.1) * r;

    // Calculate: Vext_sparse=(spdiags(Vext, 0, g3, g3));
    Eigen::MatrixXd
            dense_identity_matrix_cubed(
                number_of_grid_points_cubed,
                number_of_grid_points_cubed);

    std::cout << "Distance from centre:" << std::endl;
    std::cout << r << std::endl;
    std::cout << "External potential:" << std::endl;
    std::cout << v_ext << std::endl;
    std::cout << "Dense identy matrix cubed:" << std::endl;
    std::cout << dense_identity_matrix_cubed << std::endl;


//    Eigen::

//    dense_identity_matrix_cubed = (v_ext)*(dense_identity_matrix_cubed);


//****************************************************************************80
// Kinetic energy.
//****************************************************************************80

    double *dense_laplacian_array;

    // Use the discrete Laplacian class.
    dense_laplacian_array = l1dd ( number_of_grid_points, grid_spacing );
//    r8mat_print ( number_of_grid_points, number_of_grid_points,
//                  dense_laplacian_array, "  L1DD:" );

    // Map an STL array to an Eigen matrix.
    Eigen::Map<Eigen::MatrixXd>
            dense_laplacian_matrix(dense_laplacian_array,number_of_grid_points,
               number_of_grid_points);

//    std::cout << "Dense Laplacian matrix: " << std::endl;
//    std::cout << dense_laplacian_matrix << std::endl;

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

//    std::cout << "Sparse identy matrix: " << std::endl;
//    std::cout << sparse_identity_matrix << std::endl;

//    std::cout << "Sparse Laplaciam matrices: " << std::endl;
//    std::cout << sparse_laplacian_matrix << std::endl;

//    std::cout << "Dense identiy matrix cubed: " << std::endl;
//    std::cout << dense_identity_matrix_cubed << std::endl;

    // Calculate the kronecker product of the laplacian matrices.
    // L3=kron(kron(L,I),I)+kron(kron(I,L),I)+kron(kron(I,I),L);
    // There must be a way to do this simpler...

    Eigen::SparseMatrix<double> sparse_laplacian_matrix_term1;
    Eigen::SparseMatrix<double> sparse_laplacian_matrix_term2;
    Eigen::SparseMatrix<double> sparse_laplacian_matrix_term3;
    Eigen::SparseMatrix<double> sparse_laplacian_matrix_sum;

    sparse_laplacian_matrix_term1 =
            Eigen::kroneckerProduct(sparse_laplacian_matrix,sparse_identity_matrix);
    sparse_laplacian_matrix_term1 =
            kroneckerProduct(sparse_laplacian_matrix_term1,
                             sparse_identity_matrix).eval();
    sparse_laplacian_matrix_term2 =
            Eigen::kroneckerProduct(sparse_identity_matrix,
                                    sparse_laplacian_matrix);
    sparse_laplacian_matrix_term2 =
            Eigen::kroneckerProduct(sparse_laplacian_matrix_term2,
                             sparse_identity_matrix).eval();
    sparse_laplacian_matrix_term3 =
            Eigen::kroneckerProduct(sparse_identity_matrix,
                                    sparse_identity_matrix);
    sparse_laplacian_matrix_term3 =
            Eigen::kroneckerProduct(sparse_laplacian_matrix_term3,
                             sparse_laplacian_matrix).eval();

    sparse_laplacian_matrix_sum =
            sparse_laplacian_matrix_term1 + sparse_laplacian_matrix_term2 +
            sparse_laplacian_matrix_term3;

//    std::cout << "Sum of the sparse Laplaciam matrices: " << std::endl;
//    std::cout << sparse_laplacian_matrix_sum << std::endl;

    return 0;
}

//// Convert a 2 dim. matrix to a vector.
//Eigen::VectorXd convertTwoDimMatrixToVector(Eigen::MatrixXd matrix)
//{
//    double *p = matrix.num  ptr<double>(0); // Pointer to row 0 of X matrix
//    // Construct a vector using a pointer.
//    std::vector<double> vector(p, p+matrix.cols*matrix.rows); // Is this dangerous!?

//    return vector;
//}
