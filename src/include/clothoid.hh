#ifndef CLOTHOIDSG1_CLOTHOID_HH
#define CLOTHOIDSG1_CLOTHOID_HH

#include <curve.hh>

template<class T1>
class Clothoid : public Curve<T1>{
private:

public:
  Clothoid() : Curve<T1>() {}
  Clothoid(Configuration2<T1> ci, Configuration2<T1> cf, LEN_T l=0, K_T k=1) : Curve<T1>(ci, cf, l, k) {}
};

#endif //CLOTHOIDSG1_CLOTHOID_HH
