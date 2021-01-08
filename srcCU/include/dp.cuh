#ifndef DP_HH
#define DP_HH

#include <settings.hh>
#include <utils.cuh>
#include <typedefs.hh>
#include <configuration.cuh>
#include <dubins.cuh>
#include <constants.cuh>

#include <iostream>
#include <set>
#include <cmath>
#include <vector>
#include <sstream>
#include <algorithm>

__global__ void printResults(real_type* results, size_t discr, size_t size);

namespace DP {
  namespace {
    class Cell { //TODO please change name
    private:
      Angle _th;
      LEN_T _l; //Length of the curve
      uint _nextID;
      int _i, _j, _id;

    public:
      real_type* _results;
      Cell() : _th(ANGLE::INVALID), _l(std::numeric_limits<LEN_T>::max()), _nextID(0) {}

      BOTH Cell(Angle th, LEN_T l, uint nextID, int i=0, int j=0, int id=0) 
        : 
          _th(th), 
          _l(l),  
          _nextID(nextID), 
          _i(i), _j(j), _id(id) 
        {}

      BOTH Angle th()                 const { return this->_th; }
      BOTH LEN_T l()                  const { return this->_l; }   
      BOTH uint next()                const { return this->_nextID; }

      BOTH int i()                    const { return this->_i; }
      BOTH int j()                    const { return this->_j; }
      BOTH int id()                   const { return this->_id; }

      BOTH Angle th(Angle th) {
        this->_th = th;
        return this->th();
      }

      BOTH LEN_T l(LEN_T l) {
        this->_l = l;
        return this->l();
      }

      BOTH uint next(uint nextID){
        this->_nextID = nextID;
        return this->next();
      }

      BOTH Cell copy(const Cell &d) {
        this->th(d.th());
        this->l(d.l());
        this->next(d.next());
        this->_i=d.i();
        this->_j=d.j();
        this->_id=d.id();
        
        return *this;
      }

      BOTH Cell operator=(const Cell &d) {
        return copy(d);
      }

      std::stringstream to_string(bool pretty = false) const {
        std::stringstream out;
        if (pretty) {
          out << "th: " << this->th() << " l: " << this->l();
        } else {
          out << "<" << (Angle) (this->th() * 1.0) << ", " << (LEN_T) (this->l()) << " (" << this->_i << ", " << this->_j << ", " << this->_id << ")" << ">";
        }
        return out;
      }

      friend std::ostream &operator<<(std::ostream &out, const Cell &data) {
        out << data.to_string().str();
        return out;
      }
      
      BOTH operator real_type(){ //TODO remove this function when finished debugging
        real_type val=this->l();
        if (false) val=this->th();
        return (val>1000000 ? 999999 : val);
      }
    };
  } //Anonymous namespace to hide information

  uint guessInitialAngles(std::vector<std::set<Angle> >& moreAngles, const std::vector<Configuration2<double> >& points, const std::vector<bool> fixedAngles,  const real_type K);
  
  /* Templated functions in dp.cut
   * std::vector<Angle> bestAngles(DP::Cell* matrix, int discr, int size);
   * std::vector<Angle> solveDP (std::vector<Configuration2<double> > points, int discr, const std::vector<bool> fixedAngles, real_type* params, bool guessInitialAnglesVal=false);
   * __global__ void solveCell(DP::Cell* matrix, uint discr, uint size, std::vector<bool> fixedAngles, Configuration2<real_type>& c0, Configuration2<real_type>& c1, int& bestK, LEN_T& bestL, Angle& bestA, Angle& a00, Angle& a01, real_type* params, int i);
   */ 

  #include<dp.cut>

  template<class CURVE>
  void solveDP(std::vector<Configuration2<real_type> > points, int discr, const std::vector<bool> fixedAngles, std::vector<real_type> params, short type=2, bool guessInitialAnglesVal=false, uint nIter=1, Angle _fullAngle=m_2pi){
    Angle fullAngle=_fullAngle;
    std::vector<Angle> angles; 
    //This doesn't work for reasons I don't know
    //std::vector<Angle>(*func)(std::vector<Configuration2<real_type> > points, uint discr, const std::vector<bool> fixedAngles, std::vector<real_type> params, Angle fullAngle, bool halveDiscr, bool guessInitialAnglesVal)=NULL;
    for(uint i=0; i<nIter+1; ++i){
      switch(type){
        case 0: {
          angles=DP::solveDPFirstVersion<CURVE>(points, discr, fixedAngles, params, fullAngle, (i==0 ? false : true), guessInitialAnglesVal);
          break;
        }
        case 1:{
          angles=DP::solveDPMatrix<CURVE>(points, discr, fixedAngles, params, fullAngle, (i==0 ? false : true), guessInitialAnglesVal);
          break;
        }
        case 2: default:{
          angles=DP::solveDPAllIn1<CURVE>(points, discr, fixedAngles, params, fullAngle, (i==0 ? false : true), guessInitialAnglesVal);
        }
      }

      for (uint j=0; j<angles.size(); j++){
        if (!fixedAngles[j]){
          points[j].th(angles[j]);
        }
      }

      fullAngle=3.0*fullAngle/(2.0*discr);
    }
  }

} //namespace DP


#endif //DP_HH

















