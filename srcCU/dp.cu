#ifdef CUDA_ON
#include <dp.cuh>

// returns (up to) two circles through two points, given the radius
// Marco Frego and Paolo Bevilacqua in "An Iterative Dynamic Programming Approach to the Multipoint Markov-Dubins Problem" 2020.
static inline
void circles(real_type x1, real_type y1, real_type x2, real_type y2, real_type r, std::vector<real_type> & XC, std::vector<real_type> & YC) 
{
  real_type TOL = 1e-8;
  
  real_type q = std::hypot(x2-x1, y2-y1);
  real_type x3 = 0.5*(x1+x2);
  real_type y3 = 0.5*(y1+y2);

  real_type delta = r*r-q*q/4.;
    
  XC.clear();
  YC.clear();

  if (delta < -TOL) {
    return;
  }
  
  if (delta < TOL) 
  {
    XC.push_back(x3);
    YC.push_back(y3);
  }
  else
  {
    real_type deltaS = std::sqrt(delta);
    XC.push_back(x3 + deltaS*(y1-y2)/q);
    YC.push_back(y3 + deltaS*(x2-x1)/q);
    XC.push_back(x3 - deltaS*(y1-y2)/q);
    YC.push_back(y3 - deltaS*(x2-x1)/q);
  }
}

// Marco Frego and Paolo Bevilacqua in "An Iterative Dynamic Programming Approach to the Multipoint Markov-Dubins Problem" 2020.
uint DP::guessInitialAngles(std::vector<std::set<Angle> >& moreAngles, const std::vector<Configuration2<real_type> >& points, const std::vector<bool> fixedAngles, const real_type K){
  uint max=0;
  for (uint i=1; i<points.size(); i++){
    //std::cout << points[i-1] << "   " << points[i] << std::endl;
    moreAngles.push_back(std::set<Angle>());
    if (i==1) { moreAngles.push_back(std::set<Angle>()); }
    //First add the lines connecting two points:
    Angle th = std::atan2((points[i].y()-points[i-1].y()), (points[i].x()-points[i-1].x()));
    if (!fixedAngles[i-1]){ moreAngles[i-1].insert(th); }
    if (!fixedAngles[i])  { moreAngles[i].insert(th); }
    
    //Then add the possible angles of the tangents to two possible circles:
    std::vector<real_type> XC, YC;
    circles(points[i-1].x(), points[i-1].y(), points[i].x(), points[i].y(), 1./K, XC, YC);
    //std::cout << "aligned: " << th << std::endl;
    
    //std::cout << "XC: ";
    //for (auto a : XC) { std::cout << a << ", "; }
    //std::cout << std::endl << "YC: ";
    //for (auto a : YC) { std::cout << a << ", "; }
    for (uint j=0; j<XC.size(); j++){
      if (!fixedAngles[i-1]){
        th = std::atan2(points[i-1].y()-YC[j], points[i-1].x()-XC[j]);
        moreAngles[i-1].insert(th+M_PI/2.);
        moreAngles[i-1].insert(th-M_PI/2.);
      }
      if (!fixedAngles[i]){
        th = std::atan2(points[i].y()-YC[j], points[i].x()-XC[j]);
        moreAngles[i].insert(th+M_PI/2.);
        moreAngles[i].insert(th-M_PI/2.);
      }
    }
    if (moreAngles[i-1].size()>max){
      max=moreAngles[i-1].size();
    }
    if (i==points.size()-1 && moreAngles[i].size()>max){
      max=moreAngles[i].size();
    }
    //std::cout << std::endl << std::endl << std::endl;
  }  
  //for (auto v : moreAngles){
  //  for (auto a : v){
  //    std::cout << std::setw(11) << std::setprecision(5) << a*360/(2*M_PI) << " ";
  //  }
  //  std::cout << endl;
  //}

  //std::cout << "points: " << points.size() << " angles: " << moreAngles.size() << std::endl; 
  return max;
}

__global__ void printResults(real_type* results, size_t discr, size_t size){
  for (int i=0; i<size; i++){
    for(int j=0; j<discr; j++){
      for(int h=0; h<discr; h++){
        printf("(%2.0f,%.2f)", (float)((i*discr+j)*discr+h), results[(i*discr+j)*discr+h]);
      }
      printf("\t");
    }
    printf("\n");
  }
}

#endif 

