#include <dp.hh>

vector<Angle> bestAngles (DP::Cell** matrix, int rows, int cols, int discr){
  int id=0;
  double bL=0.0;
  for (int i=0; i<rows; i++){
    if (bL>matrix[i][1].l() || bL==0){
      bL=matrix[i][1].l();
      id=i;
    }
  }
  printf("Best length: %.8f\n", matrix[id][0].l());

  vector<Angle> ret(1,matrix[id][0].th());
  for (int i=1; i<cols; i++){
//    ret.push_back(matrix[id][i].th());
    ret.push_back(matrix[id][i].th());
    id=DP::closestDiscr(ret.back(), discr);
  }
  return ret;
}

void DP::solveDP(vector<Configuration2<double> > points, int size, int discr){ //TODO this could be parametrized.
//  DP::Cell matrix[discr][size-1]; //TODO can this be changed to a vector???
  DP::Cell** matrix=new DP::Cell*[discr];
  for (int i=0; i<discr; i++){
    matrix[i]=new DP::Cell [size]; //TODO one column could be removed but indexes need to be changed. The angle of the first point it's just m_2pi*i/discr
  }
  double L=0;
  int i=0;
  cout << m_2pi/(discr*1.0) << endl;
  for (i=points.size()-1; i>0; i--){
    Configuration2<double>* c0=&points[i-1];
    Configuration2<double>* c1=&points[i];
    for (int j=0; j<discr; j++) {
      double th0=m_2pi*(j*1.0)/(discr*1.0);
      LEN_T cL=0.0, bL=0.0;
      Angle bA=0.0;
      c0->th(th0);
      for (int k=0; k<discr; k++){
        double th1=m_2pi*(k*1.0)/(discr*1.0);
        c1->th(th1);
        CURVE c;
        try{
          c=CURVE (*c0, *c1);
        } catch (runtime_error e){}

        COUT(*c0 << " " << *c1 << " " << c.l() << endl)

        if (c.l()>0) {
          cL = c.l();
        }
        if (i!=(points.size()-1)) {
          cL+=matrix[closestDiscr(th1, discr)][i+1].l();
        }
        if (cL>0 && (cL<bL || bL==0)){
          bA=th1;
          bL=cL;
        }
      }
      COUT("i: " << i << " j: " << j << " bL: " << bL << " bA: " << bA << endl)
      matrix[j][i].l(bL);
      matrix[j][i].th(bA);
      if (i==1){ //Add values for first point.
        matrix[j][i-1].l(bL);
        matrix[j][i-1].th(th0);
      }
    }
  }
#ifdef DEBUG
  cout << "Printing " << endl;
  printM(matrix, discr, size)
#endif

  cout << "Solving" << endl;
  vector<Angle> angles=bestAngles(matrix, discr, size, discr);

  cout << "Printing for Matlab" << endl;
  cout << "x0 = " << points[0].x() << ";" << endl;
  cout << "y0 = " << points[0].y() << ";" << endl;
  cout << "th0 = " << angles[0] << ";" << endl;
  cout << "points=[";
  for (int i=1; i<size; i++){
    cout << "[" << points[i].x() << "," << points[i].y() << "," << angles[i] << "],";
  }
  cout << "];" << endl;

  double Length=0.0;
  for (int i=0; i<angles.size()-1; i++){
    points[i].th(angles[i]);
    points[i+1].th(angles[i+1]);
    Clothoid<double> c(points[i], points[i+1]);
    Length+=c.l();
  }
  cout << "Length: " << Length << endl;

  for (int i=0; i<discr; i++) {
    delete[] matrix[i];
  }
  delete [] matrix;
}

