#ifndef CLOTHOIDSG1_DUBINS_HH
#define CLOTHOIDSG1_DUBINS_HH

#import <curve.hh>

template<class T1>
class Dubins : public Curve<T1> {
private:
  K_T _kmax;

public:
  Dubins() : Curve<T1>(CURVE_TYPE::DUBINS), _kmax(0) {}
  Dubins(Configuration2<T1> ci, Configuration2<T1> cf, K_T kmax=0) : Curve<T1>(ci, cf, CURVE_TYPE::DUBINS), _kmax(kmax) {}

  K_T kmax () const { return this->_kmax; }

  K_T kmax (K_T kmax) { this->_kmax=kmax; return this->_kmax; }
};

#endif //CLOTHOIDSG1_DUBINS_HH
