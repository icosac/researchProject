#include<iostream>
#include<math.h>
using namespace std;

//#define DEBUG

#include<utils.cuh>
#include<dubins.cuh>
#include<dp.cuh>
#include<timeperf.hh>
#include<constants.cuh>

#include<tests.hh>

vector<vector<Configuration2<double> > > Tests = {
  kaya1, kaya2, kaya3, kaya4, omega, albert
};

vector<K_T> Ks = {3.0, 3.0, 5.0, 3.0, 3.0, 0.1};
vector<uint> discrs = {4, 120, 360, 720, 1440, 2880};

#define DISCR 2880

int main (){
  cout << "CUDA" << endl;
  cudaFree(0);
#if false
  for (uint discr : discrs){
    cout << "Discr: " << discr << endl;
    for (uint j=0; j<Tests.size(); j++){
      std::vector<bool> fixedAngles;
      vector<Configuration2<double> > v=Tests[j];
      for (int i=0; i<v.size(); i++){
        if (i==0 || i==v.size()-1) {
          fixedAngles.push_back(true);
        }
        else {
          fixedAngles.push_back(false);
        }
      }
      std::vector<real_type> curveParamV={Ks[j]};
      real_type* curveParam=curveParamV.data();

      TimePerf tp;
      tp.start();
      cout << "\t";
      DP::solveDP<Dubins<double> >(v, discr, fixedAngles, curveParamV, true);
      auto time=tp.getTime();
      cout << "\tExample " << j+1 << " completed in " << time << " ms" << endl;
    }
  }
  
#else
  #define KAYA omega
  std::vector<bool> fixedAngles;
  for (int i=0; i<KAYA.size(); i++){
    if (i==0 || i==KAYA.size()-1) {
      fixedAngles.push_back(true);
    }
    else {
      fixedAngles.push_back(false);
    }
  }
  std::vector<real_type> curveParamV={3.0};
  real_type* curveParam=curveParamV.data();
  TimePerf tp;
  tp.start();
  DP::solveDP<Dubins<double> >(KAYA, DISCR, fixedAngles, curveParamV, true);
  auto time=tp.getTime();
  cout << "Elapsed: " << time << " ms" << endl;

#endif
  return 0;
}

