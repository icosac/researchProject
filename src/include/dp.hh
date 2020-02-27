#ifndef CLOTHOIDSG1_DP_HH
#define CLOTHOIDSG1_DP_HH

#include <utils.hh>
#include <typedefs.hh>
#include <configuration.hh>
#include <clothoid.hh>

#include <iostream>
#include <cmath>
#include <vector>

namespace DP {
  namespace {
    class Cell { //TODO please change name
    private:
      Angle _th;
      LEN_T _l; //Length of the curve

    public:
      Cell() : _th(ANGLE::INVALID), _l(0) {}

      Cell(Angle th, LEN_T l) : _th(th), _l(l) {}

      Angle th() const { return this->_th; }

      LEN_T l() const { return this->_l; }

      Angle th(Angle th) {
        this->_th = th;
        return this->th();
      }

      LEN_T l(LEN_T l) {
        this->_l = l;
        return this->l();
      }

      Cell copy(const Cell &d) {
        this->th(d.th());
        this->l(d.l());
        return *this;
      }

      Cell operator=(const Cell &d) {
        return copy(d);
      }

      std::stringstream to_string(bool pretty = false) const {
        std::stringstream out;
        if (pretty) {
          out << "th: " << this->th() << " l: " << this->l();
        } else {
          out << "<" << (Angle) (this->th() * 1.0) << ", " << (LEN_T) (this->l()) << ">";
        }
        return out;
      }

      friend std::ostream &operator<<(std::ostream &out, const Cell &data) {
        out << data.to_string().str();
        return out;
      }
    };

    inline int closestDiscr(Angle th, int DISCR) {
      double app = th / (2 * m_pi) * DISCR;
      int a = (int) (app + 1.0e-5);
      int b = app;
      return (a == b ? b : a);
    }
  }//Anonymous namespace to hide information

  void solveDP(std::vector<Configuration2<double> > points, int size, int discr);
} //namespace DP
#endif //CLOTHOIDSG1_DP_HH
