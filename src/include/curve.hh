#ifndef CLOTHOIDSG1_CURVE_HH
#define CLOTHOIDSG1_CURVE_HH

#include <configuration.hh>

/*!
 * Type-parameter for the `Configuration2`, i.e., specifies the coordinate's type.
 * @tparam T1
 */
template<class T1>
class Curve{
private:
  Configuration2<T1> _ci; ///<Initial `Configuration`
  Configuration2<T1> _cf; ///<Final `Configuration`
  LEN_T _l;               ///<Length of the curve

public:
  /*!
   * @brief Void constructor.
   */
  Curve() : _ci(), _cf(), _l(0) {}
  /*!
   * @brief Constructor that takes two `Configuration2` and a length.
   * @param ci Initial configuration.
   * @param cf Final configuration.
   * @param l Length of the curve.
   */
  Curve(Configuration2<T1> ci, Configuration2<T1> cf, LEN_T l=0) : _ci(ci), _cf(cf), _l(l) {}

  Configuration2<T1>* ci() { return &this->_ci; } ///< Returns a pointer to the initial `Configuration2`.
  Configuration2<T1>* cf() { return &this->_cf; } ///< Returns a pointer to the final `Configuration2`.
  LEN_T l() { return this->_l; }                  ///< Returns the length of the curve.

  LEN_T l(LEN_T l) { this->_l=l; return this->l(); } ///< Sets and returns the length of the curve.
};

#endif //CLOTHOIDSG1_CURVE_HH
