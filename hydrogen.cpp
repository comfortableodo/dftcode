#include <opencv2/core/core.hpp>
#include <iostream>
#include <cmath>
#include <vector>

//using namespace std;

std::vector<double> linspace(double min, double max, int n);
static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, const cv::Mat &zgv,
              cv::Mat_<double> &X, cv::Mat_<double> &Y, cv::Mat_<double> &Z);
static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
                         const cv::Range &zgv, cv::Mat_<double> &X,
                         cv::Mat_<double> &Y, cv::Mat_<double> &Z);

int main (int argc, char* argv[])
{

    // Variable declaration.
    int number_of_grid_points;
    int number_of_grid_points_cubed;
    double grid_spacing;
    int dimensions;

    float r; // Distance from the centre.
    float v_ext; // External potential.

    dimensions = 3;
    number_of_grid_points = 30;
    number_of_grid_points_cubed = pow (number_of_grid_points, dimensions);
    v_ext = 1./r;

    std::vector<double> grid_points = linspace(-5, 5, number_of_grid_points);
    grid_spacing = grid_points[2]-grid_points[1];


//    Debugging.
//    Print out the linspace p.
    for (std::vector<double>::const_iterator i = grid_points.begin(); i != grid_points.end(); ++i)
        std::cout << *i << ' ';

    std::cout << std::endl;
    std::cout << grid_spacing << std::endl;

//    Meshgrid test
    cv::Mat_<double> X, Y, Z;
//    The class cv::range is used to specify a row or a column span in a matrix
//    ( Mat ) and for many other purposes. Range(a,b) is basically the same as
//    a:b in  Matlab or a..b in Python.
//    meshgridTest(cv::Range(1,3), cv::Range(1,3), cv::Range(1,3), X, Y, Z);
    meshgrid(cv::Mat(grid_points), cv::Mat(grid_points), cv::Mat(grid_points), X, Y, Z);
    std::cout << X << std::endl;
    std::cout << Y << std::endl;
    std::cout << Z << std::endl;



    return 0;
}

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
