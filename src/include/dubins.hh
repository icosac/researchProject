#ifndef CLOTHOIDSG1_DUBINS_HH
#define CLOTHOIDSG1_DUBINS_HH

#include <curve.hh>
#include <utils.hh>
#include <settings.hh>

#define KMAX 10

#include <cmath>
#include <limits>

template<class T1>
class Dubins : public Curve<T1> {
public:
  enum D_TYPE {INVALID, LSL, RSR, LSR, RSL, RLR, LRL};
private:
  K_T _kmax=KMAX, _k1=0.0, _k2=0.0, _k3=0.0;
  LEN_T _s1=0.0, _s2=0.0, _s3=0.0;
  D_TYPE _type;

public:
  Dubins() :
          Curve<T1>(CURVE_TYPE::DUBINS),
          _kmax(0),
          _type (D_TYPE::INVALID) {}

  Dubins(Configuration2<T1> ci, Configuration2<T1> cf, K_T kmax = 0) :
          Curve<T1>(ci, cf, CURVE_TYPE::DUBINS),
          _kmax(kmax),
          _type(D_TYPE::INVALID) {
    real_type lambda;
    K_T sKmax;
    Angle phi, sth0, sth1;

    scaleToStandard(phi, lambda, sth0, sth1, sKmax);
    computeBest(sth0, sth1, lambda);
  }

  K_T kmax() const { return this->_kmax; }
  K_T k1() const { return this->_k1; }
  K_T k2() const { return this->_k2; }
  K_T k3() const { return this->_k3; }
  LEN_T s1() const { return this->_s1; }
  LEN_T s2() const { return this->_s2; }
  LEN_T s3() const { return this->_s3; }
  LEN_T l() const { return (this->s1()+this->s2()+this->s3()); }
  D_TYPE type() const { return this->_type; }

  K_T kmax(K_T kmax) { this->_kmax = kmax; return this->kmax(); }
  K_T k1(K_T k1) { this->_k1 = k1; return this->k1(); }
  K_T k2(K_T k2) { this->_k2 = k2; return this->k2(); }
  K_T k3(K_T k3) { this->_k3 = k3; return this->k3(); }
  LEN_T s1(LEN_T s1) { this->_s1 = s1; return this->s1(); }
  LEN_T s2(LEN_T s2) { this->_s2 = s2; return this->s2(); }
  LEN_T s3(LEN_T s3) { this->_s3 = s3; return this->s3(); }
  D_TYPE type(D_TYPE type) { this->_type = type; return this->type(); }

  void scaleToStandard(Angle& phi, real_type& lambda, Angle& sth0, Angle& sth1, K_T& sKmax);
  void computeBest( Angle th0, Angle th1, real_type lambda);

};

#include <Dubins.tt>

#endif //CLOTHOIDSG1_DUBINS_HH
