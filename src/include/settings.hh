#ifndef SETTINGS_HH
#define SETTINGS_HH

#include <clothoid.hh>
#include <dubins.hh>

//#define CLOTHOID
#define DUBINS

#if defined(CLOTHOID)
typedef Clothoid<double> CURVE;
#elif defined(DUBINS)
typedef Dubins<double> CURVE;
#endif

#endif //SETTINGS_HH


