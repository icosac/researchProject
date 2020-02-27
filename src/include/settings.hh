#ifndef CLOTHOIDSG1_SETTINGS_HH
#define CLOTHOIDSG1_SETTINGS_HH

#include <clothoid.hh>
#include <dubins.hh>

#define CLOTHOID
//#define DUBINS

#if defined(CLOTHOID)
typedef Clothoid<double> CURVE;
#elif defined(DUBINS)
typedef Dubins<double> CURVE;
#endif

#endif //CLOTHOIDSG1_SETTINGS_HH


