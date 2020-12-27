#ifndef DP_HH
#define DP_HH

#include <settings.hh>
#include <utils.cuh>
#include <typedefs.hh>
#include <configuration.cuh>
#include <dubins.cuh>
#include <constants.cuh>

#include <iostream>
#include <cmath>
#include <vector>
#include <sstream>
#include <algorithm>
//#include <cmath>

__global__ void printResults(real_type* results, size_t discr, size_t size);

namespace DP {
  namespace {
    class Cell { //TODO please change name
    private:
      Angle _th;
      LEN_T _l; //Length of the curve
      uint _nextID;
      int _i, _j, _id;
      size_t _size;

    public:
      real_type* _results;
      Cell() : _th(ANGLE::INVALID), _l(std::numeric_limits<LEN_T>::max()), _nextID(0) {}

      BOTH Cell(Angle th, LEN_T l, uint nextID, 
                size_t size=0, int i=0, int j=0, int id=0) 
        : 
          _th(th), 
          _l(l),  
          _nextID(nextID), 
          _size(size), 
          _i(i), _j(j), _id(id) 
        {
        //if (this->_size!=0){
        //  #ifndef __CUDA_ARCH__
        //  cudaMallocManaged(&this->_results, sizeof(real_type)*this->_size);
        //  #endif
        //}
      }

      //BOTH ~Cell(){
      //  if (this->_size!=0) {
      //    free(this->_results);
      //  }
      //}

      BOTH Angle th()                 const { return this->_th; }

      BOTH LEN_T l()                  const { return this->_l; }
      
      BOTH uint next()                const { return this->_nextID; }
      BOTH size_t size()              const { return this->_size; }
      BOTH real_type results(int id)  const { return this->_results[id]; }
      BOTH real_type* results()             { return this->_results; }

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

      BOTH void addResult(int id, real_type value){
        if (this->size()>id) {this->_results[id]=value;}
      }

      BOTH Cell copy(const Cell &d) {
        this->th(d.th());
        this->l(d.l());
        this->next(d.next());
        this->_i=d.i();
        this->_j=d.j();
        this->_id=d.id();
        this->_size=d.size();
        this->_results=(real_type*) malloc(sizeof(real_type)*this->_size);
        for (int h=0; h<this->size(); h++){
          this->_results[h]==d.results(h);
        }

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

  void guessInitialAngles(std::vector<Configuration2<double> >& points, const std::vector<bool> fixedAngles);
  
  /* Templated functions in dp.cut
   * std::vector<Angle> bestAngles(DP::Cell* matrix, int discr, int size);
   * std::vector<Angle> solveDP (std::vector<Configuration2<double> > points, int discr, const std::vector<bool> fixedAngles, real_type* params, bool guessInitialAnglesVal=false);
   * __global__ void solveCell(DP::Cell* matrix, uint discr, uint size, std::vector<bool> fixedAngles, Configuration2<real_type>& c0, Configuration2<real_type>& c1, int& bestK, LEN_T& bestL, Angle& bestA, Angle& a00, Angle& a01, real_type* params, int i);
   */ 

  #include<dp.cut>

} //namespace DP


#endif //DP_HH

















