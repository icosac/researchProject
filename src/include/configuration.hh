#ifndef CLOTHOIDSG1_CONFIGURATION_HH
#define CLOTHOIDSG1_CONFIGURATION_HH

typedef float Angle;

template<class T1=float>
class Configuration2{
private:
  T1 _x, _y;  ///<Coordinates
  Angle _th; ///<Angle

public:
  Configuration2() : _x(0), _y(0), _th(0) {}
  Configuration2(T1 x, T1 y, Angle th) : _x(x), _y(y), _th(th) {}

  const T1 x() { return this->_x; }
  const T1 y() { return this->_y; }
  const Angle th() { return this->_th; }

  template<class T2>
  void x(const T2 x) { this->_x=((T1)x); }
  template<class T2>
  void y(const T2 y) { this->_y=((T1)y); }
  void th(const Angle th) { this->_th=th; }

  template<class T2>
  Configuration2 copy (Configuration2<T2> conf){
    this->x(conf.x());
    this->y(conf.y());
    this->th(conf.th());
    return *this;
  }

  template<class T2>
  Configuration2 operator= (Configuration2<T2> conf){
    return copy(conf);
  }
};

#endif //CLOTHOIDSG1_CONFIGURATION_HH
