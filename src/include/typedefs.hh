#ifndef TYPEDEFS_HH
#define TYPEDEFS_HH

//General
typedef double real_type; ///<Typedef to describe the real type
typedef int int_type;     ///<Typedef to describe integers

//configuration.hh
typedef float Angle;

//curve.hh
typedef double LEN_T;///<Typedef to describe the length
typedef double K_T;  ///<Typedef to describe the curvature
namespace {
  enum CURVE_TYPE { INVALID, CLOTHOID, DUBINS, DUBINS_ARC }; ///< Possible types of CURVE
}

enum ANGLE { INVALID = -1 };

#endif //TYPEDEFS_HH
