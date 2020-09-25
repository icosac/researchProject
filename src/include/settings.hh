#ifndef SETTINGS_HH
#define SETTINGS_HH

#include <clothoidG1.hh>
#include <dubins.hh>

//#define CLOTHOID
#define DUBINS

#if defined(CLOTHOID)
typedef ClothoidG1<double> CURVE;
#elif defined(DUBINS)
typedef Dubins<double> CURVE;
#endif

#endif //SETTINGS_HH


