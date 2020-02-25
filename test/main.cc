#include<clothoid.hh>

#include<iostream>
#include<cmath>
#include<random>
#include<vector>
#include<utility>
#include<iomanip>
#include<algorithm>

using namespace std;

#define SIZE 4
#define DISCR 8

enum ANGLE {INVALID=-1};

vector<Configuration2<double> > points={
//        Configuration2<double> (1, 1, 7*m_pi/16),
        Configuration2<double> (1, 1, ANGLE::INVALID),
        Configuration2<double> (3, 2, ANGLE::INVALID),
        Configuration2<double> (4, 0, ANGLE::INVALID),
        Configuration2<double> (6, 2, ANGLE::INVALID),

//        Configuration2<double> (10, 2, ANGLE::INVALID),
//        Configuration2<double> (8, 4, ANGLE::INVALID),
//        Configuration2<double> (3, 1, ANGLE::INVALID),
//        Configuration2<double> (6, 2, ANGLE::INVALID),
//        Configuration2<double> (6, 2, 15*m_pi/16),
};

class Data { //TODO please change name
private:
  Angle _th0, _th1; //exit and entrance angles respectively
  LEN_T _l; //Length of the curve

public:
  Data() : _th0(ANGLE::INVALID), _th1(ANGLE::INVALID), _l(0) {}
  Data(Angle th0, Angle th1, LEN_T l) : _th0(th0), _th1(th1), _l(l) {}

  Angle th0() const { return this->_th0; }
  Angle th1() const { return this->_th1; }
  LEN_T l() const { return this->_l; }

  Angle th0(Angle th){ this->_th0=th; return this->th0(); }
  Angle th1(Angle th){ this->_th1=th; return this->th1(); }
  LEN_T l(LEN_T l) { this->_l=l; return this->l(); }

  Data copy(const Data& d){
    this->th0(d.th0());
    this->th1(d.th1());
    this->l(d.l());
    return *this;
  }

  Data operator= (const Data& d) {
    return copy(d);
  }

  stringstream to_string(bool pretty=false) const {
    stringstream out;
    if (pretty) {
      out << "th0: " << this->th0() << " th1: " << this->th1() << " l: " << this->l();
    }
    else{
      out << "<" << (Angle)(this->th0()*1.0) << ", " << (Angle)(this->th1()*1.0) << ", " << (LEN_T)(this->l()) << ">";
    }
    return out;
  }
  friend std::ostream& operator<<(std::ostream &out, const Data& data) {
    out << data.to_string().str();
    return out;
  }
};

#include <limits>
inline int closestDiscr(Angle th0){//TODO an epsilon should be used
  double app=th0/(2*m_pi)*DISCR;
//  cout << "closestDiscrApp: " << app << endl;
  int a=(int)(app+1.0e-5);
  int b=app;
  return (a==b ? b : a);
}

bool lengthOrder(pair<LEN_T, int> a, pair<LEN_T, int> b){
  return a.first<b.first;
}

#define printV(v) \
  cout << "<"; \
  for (auto a : v) cout << a << " "; \
  cout << ">" << endl;

vector<Angle> bestAngles(Data matrix[][DISCR]){
  vector<pair<LEN_T, int> > ordered; //pair<length, id>
  vector<Angle> bestA;
  LEN_T bL=0;

  for (int i=0; i<DISCR; i++){
    if (matrix[0][i].th0()!=ANGLE::INVALID && matrix[0][i].th1()!=ANGLE::INVALID) {
      ordered.push_back(make_pair(matrix[0][i].l(), i));
    }
  }
  sort(ordered.begin(), ordered.end(), lengthOrder);

  for (int i=0; i<ordered.size(); i++) {
    LEN_T l=0;
    vector<Angle> app;
    Data d = matrix[0][ordered[i].second];
    app.push_back(d.th0());
    l+=ordered[i].first;
    for (int j=1; j<SIZE-1; j++) {
      COUT(d);
      if (d.th0()!=ANGLE::INVALID && d.th1()!=ANGLE::INVALID) {
        int id=closestDiscr(d.th1());
        COUT("Closest id of " << d.th1() << " is " << id << endl)
        d=matrix[j][closestDiscr(d.th1())];
        app.push_back(d.th0());
        if (j==SIZE-2){
          app.push_back(d.th1());
        }
        l+=d.l();
      }
      else {
        break;
      }
    }
#ifdef DEBUG
    cout << "app ";
    printV(app)
    cout << "appL=" << l << endl;
    cout << endl;
#endif
    if (l!=0 && (l<bL || bL==0)){
      bL=l;
      bestA=app;
    }
  }
  cout << "Length: " << bL << endl;
  for (auto a : bestA){
    cout << a << " ";
  }
  cout << endl;

  return bestA;
}

int main (){
  Data matrix[SIZE-1][DISCR]; //TODO can this be changed to a vector???
  double L=0;
  for (int i=points.size()-1; i>0; i--){
    Configuration2<double>* c0=&points[i-1];
    Configuration2<double>* c1=&points[i];
    int j=0; //Used to count how many th1 angles have been tested;
    for (double th0=0; th0<2*m_pi; th0+=m_pi/(DISCR*1.0/2.0)) {
      c0->th(th0);
      double bL = 0, bA = 0;
      for (double th1=0; th1<2*m_pi; th1+=m_pi/(DISCR*1.0/2.0)) {
        c1->th(th1);
        Clothoid<double> c(*c0, *c1);
        if (c.l()>0 && (c.l()<matrix[i-1][j].l() || matrix[i-1][j].l()==0)) {
          matrix[i-1][j]=(Data(th0, th1, c.l()));
          bA = th1;
          COUT("For points <" << *c.ci() << ", " << *c.cf() << ">" << "  chosen: " << bL << " " << bA << endl)
        }
      }
      j++;
    }
  }
#ifdef DEBUG
  cout << "Printing " << endl;
  for (int i=0; i<8; i++){
    for (int j=0; j<SIZE-1; j++){
      cout << matrix[j][i] << "\t";
    }
    cout << endl;
  }
#endif

  cout << "Solving" << endl;
  vector<Angle> angles=bestAngles(matrix);

  COUT("Printing for Matlab" << endl)
  COUT("x0 = " << points[0].x() << ";" << endl)
  COUT("y0 = " << points[0].y() << ";" << endl)
  COUT("th0 = " << angles[0] << ";" << endl)
  COUT("points=[")
  for (int i=1; i<SIZE; i++){
    COUT("[" << points[i].x() << "," << points[i].y() << "," << angles[i] << "],")
  }
  COUT("];" << endl)
  return 0;
}
