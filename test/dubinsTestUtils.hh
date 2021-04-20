#ifndef DUBINSTESTUTILS
#define DUBINSTESTUTILS

#define DISCR 360
#define ITER 2
#define EPSILON 1e-4

inline const char* toCString(std::stringstream msg){
  return msg.str().c_str();
}

#define READ_FROM_FILE_DUBINS()                                                                            \
  ifstream input("test/dubinsTest.txt");                                                                   \
    real_type x0, y0, th0, x1, y1, th1, kmax, l, s1, s2, s3, k1, k2, k3;                                   \
    int i=0;                                                                                               \
    while (input >> kmax >> x0 >> y0 >> th0 >> x1 >> y1 >> th1 >> l >> s1 >> s2 >> s3 >> k1 >> k2 >> k3){  \
      i++;                                                                                                 \
      Configuration2<real_type>ci(x0, y0, th0);                                                            \
      Configuration2<real_type>cf(x1, y1, th1);                                                            

#define CLOSE_FILE_DUBINS() } input.close();

#if TESTONCUDA == 0
LEN_T solve(std::vector<Configuration2<real_type> > kaya, real_type kmax){
  printf("USING THE RIGHT CODE CPU\n");
  //Compute fixedAngles, which are simply the first and the last one;
  std::vector<bool> fixedAngles;
  for (uint i=0; i<kaya.size(); i++){
    if (i==0 || i==kaya.size()-1) {
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
  std::vector<Angle> bestA=DP::solveDP<Dubins<real_type> >(kaya, DISCR, fixedAngles, params, true); //Using initial angle guess for average better results.

  //Compute total length
  LEN_T Length=0.0;
  for (uint i=0; i<bestA.size()-1; i++){
    kaya[i].th(bestA[i]);
    kaya[i+1].th(bestA[i+1]);
    Dubins<real_type> c(kaya[i], kaya[i+1], kmax);
    Length+=c.l();
  }

  free(params);
  return Length;
}
#else
LEN_T solve(std::vector<Configuration2<real_type> > kaya, real_type kmax, short dp_func=1){
  printf("USING THE RIGHT CODE GPU\n");
  //Compute fixedAngles, which are simply the first and the last one;
  std::vector<bool> fixedAngles;
  for (uint i=0; i<kaya.size(); i++){
    if (i==0 || i==kaya.size()-1) {
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
  DP::solveDP<Dubins<real_type> >(kaya, DISCR, fixedAngles, params, dp_func, true, ITER); //Using initial angle guess for average better results.

  //Compute total length
  LEN_T Length=0.0;
  for (uint i=0; i<kaya.size()-1; i++){
    Dubins<real_type> c(kaya[i], kaya[i+1], kmax);
    Length+=c.l();
  }

  return Length;
}
#endif //CUDA_ON



#endif //DUBINSTESTUTILS