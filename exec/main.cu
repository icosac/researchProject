#include<iostream>
#include<math.h>
using namespace std;

#define DEBUG

#include<utils.cuh>
#include<dubins.cuh>
#include<dp.cuh>
#include<timeperf.hh>

vector<Configuration2<double> > kaya1={
        Configuration2<double> (0, 0, -M_PI/3.0),
        Configuration2<double> (-0.1, 0.3, ANGLE::INVALID),
        Configuration2<double> (0.2, 0.8, ANGLE::INVALID),
        Configuration2<double> (1, 1, -M_PI/6.0)
};

vector<Configuration2<double> > kaya2={
        Configuration2<double> (0, 0, -M_PI/3.0),
        Configuration2<double> (-0.1, 0.3, ANGLE::INVALID),
        Configuration2<double> (0.2, 0.8, ANGLE::INVALID),
        Configuration2<double> (1, 1, ANGLE::INVALID),
        Configuration2<double> (0.5, 0.5, ANGLE::INVALID),
        Configuration2<double> (0.5, 0, -M_PI/6.0)
};

vector<Configuration2<double> > kaya4={
       Configuration2<double>(0.5, 1.2, 5*M_PI/6.0),
       Configuration2<double>(0.0, 0.5, ANGLE::INVALID),
       Configuration2<double>(0.5, 0.5, ANGLE::INVALID),
       Configuration2<double>(1.0, 0.5, ANGLE::INVALID),
       Configuration2<double>(1.5, 0.5, ANGLE::INVALID),
       Configuration2<double>(2.0, 0.5, ANGLE::INVALID),
       Configuration2<double>(2.0, 0.0, ANGLE::INVALID),
       Configuration2<double>(1.5, 0.0, ANGLE::INVALID),
       Configuration2<double>(1.0, 0.0, ANGLE::INVALID),
       Configuration2<double>(0.5, 0.0, ANGLE::INVALID),
       Configuration2<double>(0.0, 0.0, ANGLE::INVALID),
       Configuration2<double>(0.0, -0.5, 0)
};

vector<Configuration2<double> > kaya3={
       Configuration2<double>(0.5, 1.2, 5.0*M_PI/6.0),
       Configuration2<double>(0, 0.8, ANGLE::INVALID),
       Configuration2<double>(0, 0.4, ANGLE::INVALID),
       Configuration2<double>(0.1, 0, ANGLE::INVALID),
       Configuration2<double>(0.4, 0.2, ANGLE::INVALID),
       Configuration2<double>(0.5, 0.5, ANGLE::INVALID),
       Configuration2<double>(0.6, 1, ANGLE::INVALID),
       Configuration2<double>(1, 0.8, ANGLE::INVALID),
       Configuration2<double>(1, 0, ANGLE::INVALID),
       Configuration2<double>(1.4, 0.2, ANGLE::INVALID),
       Configuration2<double>(1.2, 1, ANGLE::INVALID),
       Configuration2<double>(1.5, 1.2, ANGLE::INVALID),
       Configuration2<double>(2, 1.5, ANGLE::INVALID),
       Configuration2<double>(1.5, 0.8, ANGLE::INVALID),
       Configuration2<double>(1.5, 0, ANGLE::INVALID),
       Configuration2<double>(1.7, 0.6, ANGLE::INVALID),
       Configuration2<double>(1.9, 1, ANGLE::INVALID),
       Configuration2<double>(2, 0.5, ANGLE::INVALID),
       Configuration2<double>(1.9, 0, ANGLE::INVALID),
       Configuration2<double>(2.5, 0.6, 0),
};

vector<vector<Configuration2<double> > > Tests = {
  kaya1, kaya2, kaya3, kaya4
};

vector<K_T> Ks = {3.0, 3.0, 5.0, 3.0};
vector<uint> discrs = {4, 120, 360, 720, 2000};

#define DISCR 6

int main (){
  cout << "CUDA" << endl;
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
      DP::solveDP<Dubins<double> >(v, discr, fixedAngles, curveParamV, false);
      auto time=tp.getTime();
      cout << "\tExample " << j << " completed in " << time << " ms" << endl;
    }
  }
  
#else
  #define KAYA kaya1
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

  DP::solveDP<Dubins<double> >(KAYA, DISCR, fixedAngles, curveParamV, false);
  //DP::ciao();
#endif
  return 0;
}

/*
int main(){
  cudaError_t cudaErr=cudaSuccess;

  Cell* matrix; 
  Configuration<double>* conf;
  
  cudaMallocManaged(&matrix, SIZE*sizeof(Cell));
  cudaMallocManaged(&conf, DIM*sizeof(Configuration<double>));

  for (int i=DIM; i>0; i--){
    conf[i-1].x=i;
    conf[i-1].y=i;
    conf[i-1].th=0;
  }

  for (int i=0; i<DIM; i++){
    for (int j=0; j<DISC; j++){
      matrix[i*DISC+j].val=i*DISC+j;
    }
  }
  printVM(matrix, DISC, DIM)

  for (int i=0; i<DIM; i++){
    for (int j=0; j<DISC; j++){
      matrix[i*DISC+j].val=0;
    }
  }
  
  for (int i=DIM-1; i>0; i--){
    kernel<<<1, DISC>>>(matrix, i, conf);
    cudaCheckError(cudaGetLastError());
    cudaErr=cudaDeviceSynchronize();  
    cudaCheckError(cudaErr);
    printVM(matrix, DISC, DIM)
    cout << endl;
  }

  cudaFree(matrix);
  return 0;
}
*/
