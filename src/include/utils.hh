#ifndef CLOTHOIDSG1_UTILS_HH
#define CLOTHOIDSG1_UTILS_HH

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <limits>

#include <typedefs.hh>

//#define DEBUG

#ifdef DEBUG
#define COUT(x) cout << x;
#else
#define COUT(x)
#endif

#ifndef ASSERT
  #define ASSERT(COND,MSG)         \
    if ( !(COND) ) {                        \
      std::ostringstream ost ;              \
      ost << "On line: " << __LINE__        \
          << " file: " << __FILE__          \
          << '\n' << MSG << '\n' ;          \
      throw std::runtime_error(ost.str()) ; \
    }
#endif //ASSERT

extern real_type const epsi        ;
extern real_type const m_pi        ; // pi
extern real_type const m_pi_2      ; // pi/2
extern real_type const m_2pi       ; // 2*pi
extern real_type const m_1_pi      ; // 1/pi
extern real_type const m_1_sqrt_pi ; // 1/sqrt(pi)

#define printV(v) \
  cout << "<"; \
  for (auto a : v) cout << a << " "; \
  cout << ">" << endl;

#define printM(M) \
for (int i=0; i<DISCR; i++){   \
  cout << "th" << i;    \
  for (int j=0; j<SIZE; j++){\
    cout << M[i][j] << "\t";   \
  }                            \
  cout << endl;                \
}

#endif //CLOTHOIDSG1_UTILS_HH
