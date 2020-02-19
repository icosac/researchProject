#include<iostream>
#include<clothoid.hh>

using namespace std;

int main (){
  Clothoid<int> c(Configuration2<int>(1, 2, 3), Configuration2<int>(2, 1, 3));
  cout << c.ci().x() << " " << c.cf().x() << endl;
  cout << c.ci().y() << " " << c.cf().y() << endl;
  cout << c.ci().th() << " " << c.cf().th() << endl;
  return 0;
}
