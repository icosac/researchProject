#include <dp.hh>

#define KMAX 3

void draw(CURVE & dc, AsyPlot& plot, string const & penna){
  std::stringstream str;
  
  double x0 = dc.ci()->x();
  double y0 = dc.ci()->y();
  double th0 = dc.ci()->th();
  double k0 = dc.k1();
  double s0 = dc.s1();

  double xM0 = f(s0, k0, th0) + x0;
  double yM0 = g(s0, k0, th0) + y0;
  double thetaM0 = th0 + k0*s0;
  double k1 = dc.k2();
  double s1 = dc.s2();

  double xM1 = f(s1, k1, thetaM0) + xM0;
  double yM1 = g(s1, k1, thetaM0) + yM0;
  double thetaM1 = thetaM0 + k1*s1;
  double k2 = dc.k3();
  double s2 = dc.s3();

  str		<< "path pclot = clothoidPoints(("
      << x0 << ','
      << y0 << "),"
      << th0 << ','
      << k0 << ','
      << 0 << ','
      << s0 << ','
      << "50,0);\n"
      << "pen penna = " << penna << ";\n"
      << "draw(pclot, penna);\n\n";

  str << "path pclot = clothoidPoints(("
    << xM0 << ','
    << yM0 << ")," 
    << thetaM0 << ','
    << k1 << ','
    << 0 << ','
    << s1 << ','
    << "50,0);\n"
    << "pen penna = " << penna << ";\n"
    << "draw(pclot, penna);\n\n";

  str << "path pclot = clothoidPoints(("
    << xM1 << ','
    << yM1 << "),"
    << thetaM1 << ','
    << k2 << ','
    << 0 << ','
    << s2 << ','
    << "50,0);\n"
    << "pen penna = " << penna << ";\n"
    << "draw(pclot, penna);\n\n";

  // str << "dot((" << x0 << ','
  //   << y0 << "), red+4bp);\n\n";

  // str << "dot((" << x1 << ','
  //   << y1 << "), red+4bp);\n\n";

  // str << "dot((" << xM0 << ','
  //   << yM0 << "), royalblue+2bp);\n\n";
  // str << "dot((" << xM1 << ','
  //   << yM1 << "), royalblue+2bp);\n\n";

  plot.writeLine(str.str());
  
}


void drawSolution(std::vector<double> const & x, std::vector<double> const & y, std::vector<double> const & theta, double Kmax)
{
  AsyPlot ap("dubinsDP.asy");
  for (int i=1; i<x.size(); ++i) {
    CURVE dc=CURVE(Configuration2<double>(x[i-1], y[i-1], theta[i-1]), Configuration2<double>(x[i], y[i], theta[i]), Kmax);
    draw(dc, ap, "blue");
  }

  for (int i=0; i<x.size(); ++i) {
    ap.dot(x[i], y[i], "red+2bp");
  }
}

vector<Angle> bestAnglesOld(DP::Cell** matrix, int rows, int cols, int discr){
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
    ret.push_back(matrix[id][i].th());
    id=DP::closestDiscr(ret.back(), discr);
  }
  return ret;
}

vector<Angle> bestAnglesOld(DP::Cell* matrix, int rows, int cols, int discr){


}

void DP::solveDP(vector<Configuration2<double> > points, int discr, int startFromBottom, int stopFromTop){ //TODO this could be parametrized.
  uint size=points.size();

  DP::Cell* matrix=new DP::Cell[discr*size];
  
  for (uint i=size-1; i>0; i--){
    cout << endl;
    Configuration2<double>* c0=&points[i-1];
    Configuration2<double>* c1=&points[i];
    for (int j=0; j<discr; j++){
      Angle bestA=0.0;
      LEN_T bestL=-1.0; 
      int bestK=0;
      c0->th(m_2pi*(j*1.0)/(discr*1.0));  
      for (int k=0; k<discr; k++){
        LEN_T currL=0.0;
        c1->th(m_2pi*(k*1.0)/(discr*1.0));
        CURVE c;
        try{
          c=CURVE(*c0, *c1, KMAX); 
        } catch (runtime_error e){}
        //cout << "c0: " << *c0 << " c1: " << *c1 << " l: " << c.l() << endl;
        Cell* next=(i==size-1 ? NULL : &matrix[k*size+i+1]);
        if (c.l()>0){
          currL=c.l();
          //cout << (next==NULL ? "null" : next->to_string().str()) << endl;
          if (next!=NULL){
            currL+=next->l();
          }  
          if (currL<bestL || bestL<0.0){
            bestL=currL;
            bestA=c1->th();
            bestK=k;
          }
        }
        if (j==2){
          cout << "currL: " << currL << " c.l: " << c.l() << " next: " << (next==NULL ? "NULL" : next->to_string().str()) << 
            " c0: " << *c0 << " c1: " << *c1 << endl;
        }
      }
      if (bestL>-1.0){
        Cell* next=(i==size-1 ? NULL : &matrix[bestK*size+i+1]);
        matrix[j*size+i]=Cell(bestA, bestL, next, i, j);
        //cout << "i: " << i << " j: " << j << " id: " << j*size+i << endl;
      }
      if (i==1){
        matrix[size*j]=Cell(c0->th(), bestL, &matrix[size*j+i], 0, j);
      }
    cout << endl;
    }
  }
#ifdef DEBUG
  cout << "Printing " << endl;
  printVM(matrix, discr, size)
#endif
  delete [] matrix;
}

void DP::solveDPOld(vector<Configuration2<double> > points, int size, int discr, int startFromBottom, int stopFromTop){ //TODO this could be parametrized.
//  DP::Cell matrix[discr][size-1]; //TODO can this be changed to a vector???
  DP::Cell** matrix=new DP::Cell*[discr];
  for (int i=0; i<discr; i++){
    matrix[i]=new DP::Cell [size]; //TODO one column could be removed but indexes need to be changed. The angle of the first point it's just m_2pi*i/discr
  }
  unsigned int i=0;
  const unsigned int TOP=points.size()-stopFromTop;
  for (i=points.size()-1; i>0; i--){
    Configuration2<double>* c0=&points[i-1];
    Configuration2<double>* c1=&points[i];
    Angle appAngle=c0->th();
    if (i<TOP){ //Need to do this because otherwise I don't copy the angles. Is there a better solution? 
      for (int j=0; j<discr; j++) {
        double th0=c0->th();
        if (i>startFromBottom){
          th0=m_2pi*(j*1.0)/(discr*1.0);
          c0->th(th0);
        }
        LEN_T cL=0.0, bL=0.0;
        Angle bA=0.0;
        for (int k=0; k<discr; k++){
          double th1=m_2pi*(k*1.0)/(discr*1.0);
          c1->th(th1);
          CURVE c;
          try{
            c=CURVE (*c0, *c1, KMAX);
          } catch (runtime_error e){}
//          COUT(*c0 << " " << *c1 << " " << c.l() << endl)
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
//        COUT("i: " << i << " j: " << j << " bL: " << bL << " bA: " << bA << endl)
        matrix[j][i].l(bL);
        matrix[j][i].th(bA);
        if (i==1){ //Add values for first point.
          matrix[j][i-1].l(bL);
          matrix[j][i-1].th(th0);
        }
      }
    }
    else{
      for (int j=0; j<discr; j++) {
        matrix[j][i].th(c1->th());
      }
    }
  }
#ifdef DEBUG
  cout << "Printing " << endl;
  printM(matrix, discr, size)
#endif

  cout << "Solving" << endl;
  vector<Angle> angles=bestAnglesOld(matrix, discr, size, discr);

  cout << "Printing for Matlab" << endl;
  cout << "X=[";
  for (unsigned int i=0; i<points.size(); i++){ cout << points[i].x() << (i!=points.size()-1 ? ", " : "];\n"); }
  cout << "Y=[";
  for (unsigned int i=0; i<points.size(); i++){ cout << points[i].y() << (i!=points.size()-1 ? ", " : "];\n"); }
  cout << "th=[";
  for (unsigned int i=0; i<angles.size(); i++){ cout << angles[i] << (i!=angles.size()-1 ? ", " : "];\n"); }
  //cout << "x0 = " << points[0].x() << ";" << endl;
  //cout << "y0 = " << points[0].y() << ";" << endl;
  //cout << "th0 = " << angles[0] << ";" << endl;
  //cout << "points=[";
  //for (int i=1; i<size; i++){
  //  cout << "[" << points[i].x() << "," << points[i].y() << "," << angles[i] << "],";
  //}
  //cout << "];" << endl;

  double Length=0.0;
  for (unsigned int i=0; i<angles.size()-1; i++){
    points[i].th(angles[i]);
    points[i+1].th(angles[i+1]);
    CURVE c(points[i], points[i+1], 5);
    Length+=c.l();
  }
  cout << "Length: " << Length << endl;

  cout << "Drawing" << endl;
  vector<double> X, Y, THETA;
  for (unsigned int i=0; i<points.size(); i++){ 
    X.push_back(points[i].x()); 
    Y.push_back(points[i].y()); 
    THETA.push_back(angles[i]); 
  }
  drawSolution(X, Y, THETA, KMAX);

  for (int i=0; i<discr; i++) {
    delete[] matrix[i];
  }
  delete [] matrix;
}

