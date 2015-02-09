# include <eigen3/Eigen/Dense>
# include <eigen3/Eigen/Sparse>
# include <eigen3/Eigen/StdVector>
# include <eigen3/Eigen/Eigenvalues>
# include <eigen3/unsupported/Eigen/KroneckerProduct>
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <vector>

using namespace std;

# include "laplacian.hpp"

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

//****************************************************************************80
// Templates
//****************************************************************************80

/**
 * @brief powerIteration Compute the dominant eigenvalue and its relative eigenvector of a square matrix
 * @param A The input matrix
 * @param eigenVector The eigenvector
 * @param tolerance Maximum tolerance
 * @param nIterations Number of iterations
 * @return The dominant eigenvalue
 */
template < typename Derived,  typename OtherDerived>
typename Derived::Scalar powerIteration(const Eigen::MatrixBase<Derived>& A, Eigen::MatrixBase<OtherDerived>  & eigenVector, typename Derived::Scalar tolerance,  int nIterations)
{
    typedef typename Derived::Scalar Scalar;

    OtherDerived approx(A.cols());
    approx.setRandom(A.cols());
    int counter = 0;
    Scalar error=100;
    while (counter < nIterations && error > tolerance  )
    {
        OtherDerived temp = approx;
        approx = (A*temp).normalized();
        error = (temp-approx).stableNorm();
        counter++;
    }
    eigenVector = approx;

    Scalar dominantEigenvalue = approx.transpose()*A*approx;
#ifdef INFO_LOG
    cerr << "Power Iteration:" << endl;
    cerr << "\tTotal iterations= " << counter << endl;
    cerr << "\tError= " << error << endl;
    cerr << "\tDominant Eigenvalue= " << dominantEigenvalue << endl;
    cerr << "\tDominant Eigenvector= [" << eigenVector.transpose()<< "]" << endl;
#endif
    return dominantEigenvalue;
}

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
    Eigen::ArrayXd r(number_of_grid_points_cubed); // Distance from the centre.

    Eigen::ArrayXd x2y2(number_of_grid_points*number_of_grid_points);

//    Eigen::VectorXd v_ext; // External potential.
    Eigen::ArrayXd v_ext(number_of_grid_points_cubed);


//****************************************************************************80
// External potential.
//****************************************************************************80


    // Vector containing the grid points.
    Eigen::ArrayXd grid_points(number_of_grid_points);
    grid_points.setLinSpaced(number_of_grid_points,
                             (-1)*box_side_length,
                             box_side_length);

    // Grid spacing:
    grid_spacing = grid_points[2]-grid_points[1];
    // Print out the linspace grid_points.
//    std::cout << grid_points << std::endl;

    //  Call Matlab's meshgrid function equivalent.
    // calculate d

    Eigen::ArrayXXd::Map(x2y2.data(), number_of_grid_points, number_of_grid_points) =
            grid_points.square().transpose().replicate(number_of_grid_points,1) +
            grid_points.square().replicate(1,number_of_grid_points);

    Eigen::ArrayXXd::Map(r.data(), number_of_grid_points*number_of_grid_points,
                         number_of_grid_points) =
            x2y2.replicate(1,number_of_grid_points) +
            grid_points.square().transpose().replicate(number_of_grid_points*
                                              number_of_grid_points,1);

    // sqrt(r)
    r = r.array().sqrt();

//    std::cout << "r: " << std::endl;
//    std::cout << r << std::endl;

    // r=-1/v_ext
    v_ext = (-0.1) * r;

//****************************************************************************80

    // Calculate: Vext_sparse=(spdiags(Vext, 0, g3, g3));
    Eigen::MatrixXd
            dense_identity_matrix_cubed(
                number_of_grid_points_cubed,
                number_of_grid_points_cubed);
    dense_identity_matrix_cubed.setIdentity(
                number_of_grid_points_cubed,
                number_of_grid_points_cubed);

    Eigen::MatrixXd
            dense_v_ext_matrix_cubed(
                number_of_grid_points_cubed,
                number_of_grid_points_cubed);


//    std::cout << "Distance from centre:" << std::endl;
//    std::cout << r << std::endl;
//    std::cout << "External potential:" << std::endl;
//    std::cout << v_ext << std::endl;
//    std::cout << "External potential transposed:" << std::endl;
//    std::cout << v_ext.transpose() << std::endl;
//    std::cout << "Dense identy matrix cubed:" << std::endl;
//    std::cout << dense_identity_matrix_cubed << std::endl;

    // Should probably make sparse!!
//    Eigen::SparseMatrix<double> sparse_identity_matrix_cubed =
//            dense_identity_matrix_cubed.sparseView();

    dense_v_ext_matrix_cubed = dense_identity_matrix_cubed.array().rowwise() *
            v_ext.transpose();

//    std::cout << "TEST: " << std::endl;
//    std::cout << dense_identity_matrix_cubed.array().rowwise() * v_ext.transpose() << std::endl;

    Eigen::SparseMatrix<double> sparse_v_ext_matrix_cubed =
                dense_v_ext_matrix_cubed.sparseView();

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

//****************************************************************************80
// Eigenvalues.
//****************************************************************************80

//    std::cout << "sparse_laplacian_matrix_sum.rows()" << std::endl;
//    std::cout << sparse_laplacian_matrix_sum.rows() << std::endl;
//    std::cout << "sparse_laplacian_matrix_sum.cols()" << std::endl;
//    std::cout << sparse_laplacian_matrix_sum.cols() << std::endl;
//    std::cout << "sparse_v_ext_matrx_cubed.rows()" << std::endl;
//    std::cout << sparse_v_ext_matrix_cubed.rows() << std::endl;
//    std::cout << "sparse_v_ext_matrx_cubed.cols()" << std::endl;
//    std::cout << sparse_v_ext_matrix_cubed.cols() << std::endl;

    Eigen::MatrixXd dense_laplacian_matrix_sum =
            Eigen::MatrixXd(sparse_laplacian_matrix_sum);

    // The addition of sparse matrices produced errors.
    // It is not yet clear if this is due to a bug in the library or
    // due to a wrong implementation here.
//    std::cout << "Kinetic + pot. matrix" << std::endl;
//    std::cout << (-0.5)*dense_laplacian_matrix_sum + dense_v_ext_matrix_cubed << std::endl;

    Eigen::MatrixXd hamiltonian;
    hamiltonian = (-0.5)*dense_laplacian_matrix_sum + dense_v_ext_matrix_cubed;

    // Check if there is a way to get eigenvalues of sparse matrices.
    // See MATLAB's "eigs" in regard to this.
//    Eigen::SparseMatrix<double> sparse_hamiltonian = hamiltonian.sparseView();


    Eigen::EigenSolver<Eigen::MatrixXd> es(hamiltonian);
    Eigen::VectorXd v = es.eigenvectors();

    Eigen::MatrixXd::Scalar tolerance = 0.1;
    int iterations = 10;

    Eigen::MatrixXd::Scalar test_energy =
            powerIteration(Eigen::MatrixBase<Eigen::MatrixXd> hamiltonian, Eigen::MatrixBase<Eigen::VectorXd> v, tolerance, iterations);

    std::cout << "Eigenvalues:" << std::endl;
    std::cout << test_energy << std::endl;

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
