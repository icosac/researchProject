#ifndef CLOTHOIDSG1_CURVE_HH
#define CLOTHOIDSG1_CURVE_HH

#include <configuration.hh>

template<class T1>
class Curve{
private:
  Configuration2<T1> _ci; ///<Initial `Configuration`
  Configuration2<T1> _cf; ///<Final `Configuration`
  LEN_T _l;               ///<Length of the curve

public:
  Curve() : _ci(), _cf(), _l(0) {}
  Curve(Configuration2<T1> ci, Configuration2<T1> cf, LEN_T l=0) : _ci(ci), _cf(cf), _l(l) {}

  Configuration2<T1>* ci() { return &this->_ci; }
  Configuration2<T1>* cf() { return &this->_cf; }
  LEN_T l() { return this->_l; }

  LEN_T l(LEN_T l) { this->_l=l; return this->l(); }
};

#endif //CLOTHOIDSG1_CURVE_HH
