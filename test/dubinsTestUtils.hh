#ifndef DUBINSTESTUTILS
#define DUBINSTESTUTILS

#define DISCR 360
#define ITER 2
#define EPSILON 1e-11
#define CONFIDENCE 0.04

#if !defined(DEBUG) && defined(DEBUGTEST)
  #define COUT(x) cout << x;
#endif

inline const char* toCString(std::stringstream msg){
  return msg.str().c_str();
}

#if TESTONCUDA == 0
LEN_T solve(std::vector<Configuration2<real_type> > points, real_type kmax){
  //Compute fixedAngles, which are simply the first and the last one;
  std::vector<bool> fixedAngles;
  for (uint i=0; i<points.size(); i++){
    if (i==0 || i==points.size()-1) {
      fixedAngles.push_back(true);
    }
    else {
      fixedAngles.push_back(false);
    }
  }
  //Write parameters
  real_type* params=new real_type;
  params[0]=kmax;

  //Solve DP problem
  std::vector<Angle> bestA=DP::solveDP<Dubins<real_type> >(points, DISCR, fixedAngles, params, true); //Using initial angle guess for average better results.

  //Compute total length
  LEN_T Length=0.0;
  for (uint i=0; i<bestA.size()-1; i++){
    points[i].th(bestA[i]);
    points[i+1].th(bestA[i+1]);
    Dubins<real_type> c(points[i], points[i+1], kmax);
    Length+=c.l();
  }

  free(params);
  return Length;
}

Dubins<real_type> solveDubins(Configuration2<real_type> ci, Configuration2<real_type> cf, real_type kmax){
  return Dubins<real_type>(ci, cf, kmax);
}

#else
LEN_T solve(std::vector<Configuration2<real_type> > points, real_type kmax, short dp_func=1){
  //Compute fixedAngles, which are simply the first and the last one;
  std::vector<bool> fixedAngles;
  for (uint i=0; i<points.size(); i++){
    if (i==0 || i==points.size()-1) {
      fixedAngles.push_back(true);
    }
    else {
      fixedAngles.push_back(false);
    }
  }
  //Write parameters
  std::vector<real_type> params={kmax};
  if (dp_func==2) params.push_back(3.0);

  //Solve DP problem
  DP::solveDP<Dubins<real_type> >(points, DISCR, fixedAngles, params, dp_func, true, ITER); //Using initial angle guess for average better results.

  //Compute total length
  LEN_T Length=0.0;
  for (uint i=0; i<points.size()-1; i++){
    Dubins<real_type> c(points[i], points[i+1], kmax);
    Length+=c.l();
  }

  return Length;
}

__global__ void solveDubinsKernel(Configuration2<real_type> ci, Configuration2<real_type> cf, real_type kmax, Dubins<real_type>* d){
  d[0]=Dubins<real_type>(ci, cf, kmax);
}

Dubins<real_type> solveDubins(Configuration2<real_type> ci, Configuration2<real_type> cf, real_type kmax){
  Dubins<real_type>* d;
  cudaMallocManaged(&d, sizeof(Dubins<real_type>));

  solveDubinsKernel<<<1,1>>>(ci, cf, kmax, d);
  cudaDeviceSynchronize();

  return *d;
}
#endif //TESTONCUDA


#endif //DUBINSTESTUTILS