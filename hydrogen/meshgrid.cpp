#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{

   float fSamplerate = 44100;

   double Xmic = 1;
   double Ymic = 1;
   double Zmic = 1;

   double Xsrc = 3;
   double Ysrc = 2;
   double Zsrc = 1;

   double Xrm = 5;
   double Yrm = 4;
   double Zrm = 3;

   int n = 5;
   int ntemp,icounter;
   double dTemp;
   double r = 0.4;

   Eigen::ArrayXd nn(2 * n + 1);    // nn(11)

   int N = nn.rows();               // N = 11

   Eigen::ArrayXd rms(N);
   Eigen::ArrayXd srcs(N);

   Eigen::ArrayXd xi(N);
   Eigen::ArrayXd yj(N);
   Eigen::ArrayXd zk(N);

   Eigen::ArrayXd x2y2(N*N);        // x2y2(11*11)
   Eigen::ArrayXd d(N*N*N);


   // init nn
   icounter = 0;
   for(ntemp = -n; ntemp <= n; ntemp++)
   {
      nn(icounter) = ntemp;
      icounter++;
   }

   //init rms and srcs
   for(icounter = 0; icounter <= 2*n; icounter++)
   {
      dTemp = pow(-1.0,nn(icounter));
      rms(icounter) = nn(icounter) + (0.5 - 0.5 * dTemp);
      srcs(icounter) = dTemp;
   }

   // calculate relative positions along x,y,z axis
   xi = srcs * Xsrc + (rms * Xrm) - Xmic;
   yj = srcs * Ysrc + (rms * Yrm) - Ymic;
   zk = srcs * Zsrc + (rms * Zrm) - Zmic;

   // calculate d
   ArrayXXd::Map(x2y2.data(), N, N) = xi.square().transpose().replicate(N,1) +
           yj.square().replicate(1,N);
   ArrayXXd::Map(d.data(), N*N, N) = x2y2.replicate(1,N) +
           zk.square().transpose().replicate(N*N,1);

   d = d.sqrt();

   std::cout << "d = " << d << std::endl;
}
