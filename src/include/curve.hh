#ifndef CLOTHOIDSG1_CURVE_HH
#define CLOTHOIDSG1_CURVE_HH

#include <configuration.hh>

typedef double LEN_T;///<Typedef to describe the length
typedef double K_T;  ///<Typedef to describe the curvature

template<class T1>
class Curve{
private:
  Configuration2<T1> _ci; ///<Initial `Configuration`
  Configuration2<T1> _cf; ///<Final `Configuration`
  LEN_T _l;               ///<Length of the curve
  K_T _k;                 ///<Curvature of the curve

public:
  Curve() : _ci(), _cf(), _l(0), _k(1) {}
  Curve(Configuration2<T1> ci, Configuration2<T1> cf, LEN_T l=0, K_T k=1) : _ci(ci), _cf(cf), _l(l), _k(k) {}

  Configuration2<T1> ci() { return this->_ci; }
  Configuration2<T1> cf() { return this->_cf; }
  LEN_T l() { return this->_l; }
  K_T k() { return this->_k; }
};

#endif //CLOTHOIDSG1_CURVE_HH
