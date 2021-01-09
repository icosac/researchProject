#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
#include<stdlib.h>
#include<unistd.h>
using namespace std;

// #define DEBUG

#include<utils.cuh>
#include<dubins.cuh>
#include<dp.cuh>
#include<timeperf.hh>
// #include<constants.cuh>

#include<tests.hh>

void PrintScientific1D(double d)
{
  if (d == 0)
  {
    printf ("%*d", 6, 0);
    return;
  }

  int exponent  = (int)floor(log10( fabs(d)));  // This will round down the exponent
  double base   = d * pow(10, -1.0*exponent);

  printf("%1.1lfe%+01d", base, exponent);
}

void PrintScientific2D(double d)
{
  if (d == 0)
  {
    printf ("%*d", 7, 0);
    return;
  }

  int exponent  = (int)floor(log10( fabs(d)));  // This will round down the exponent
  double base   = d * pow(10, -1.0*exponent);

  printf("%1.1lfe%+02d", base, exponent);
}

std::vector<Configuration2<real_type> > example1 = {
  Configuration2<real_type>(0,0,-2.0*M_PI/8.0),
  Configuration2<real_type>(2,2,ANGLE::INVALID),
  Configuration2<real_type>(6,-1,ANGLE::INVALID),
  Configuration2<real_type>(8,1,2.0*M_PI/8.0)
};

std::vector<std::string> testsNames = { 
  "Kaya Example 1",
  "Kaya Example 2",
  "Kaya Example 3",
  "Kaya Example 4",
  "Omega",
  "Circuit"
}; 

std::vector<std::vector<Configuration2<double> > > Tests = {
  kaya1, kaya2, kaya3, kaya4, omega, spa
};

std::vector<K_T> Ks = {3.0, 3.0, 5.0, 3.0, 3.0, 3.0};
std::vector<uint> discrs = {4, 16, 90, 360};
//std::vector<uint> discrs = {4, 16, 90, 120, 360, 720, 1440};
std::vector<uint> refins = {0, 1, 2, 4, 8, 16};
std::vector<LEN_T> exampleLenghts={3.41557885807514871601142658619, 6.27803455030931356617429628386, 11.9162126542854860389297755319, 7.46756219733842652175326293218, 41.0725016438839318766440555919, 6988.66098639942993031581863761}; //the last length is SPA

std::string nameTest(std::string name, std::string add="", std::string conc=" "){
  if (add==""){
    return name;
  }
  else{
    return name+conc+add;
  }
}

int main (int argc, char* argv[]){

  cout << "CUDA" << endl;
  cudaFree(0);

  int devicesCount;
  cudaGetDeviceCount(&devicesCount);
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, 0);
  printf("[%d] %s\n", 0, deviceProperties.name);


#if false
  std::string nExec="1";
  int jump, discrID, j;
  if (argc==2){
    nExec=std::string(argv[1]);
  }
  else if (argc==4){
    jump=atoi(argv[1]);
    discrID=atoi(argv[2]);
    j=atoi(argv[3]);
  }
  else{
    std::cout << "Many blocks activated" << std::endl;
    return 1;
  }
  int testI=0;
  // std::cout << "\t\t        \tMatrix\t\tCol\tCol-Matrix" << std::endl;
  //for (jump; jump<18; jump+=15){
    //for (discrID; discrID<discrs.size(); discrID++){
      uint discr =discrs[discrID];
      cout << "Discr: " << discr << endl;
      //for (uint j=0; j<Tests.size(); j++){
        fstream json_out; json_out.open("testResults/tests.json", std::fstream::app);
        std::vector<bool> fixedAngles;
        vector<Configuration2<double> > v=Tests[j];
        for (int i=0; i<v.size(); i++){
          if (i==0 || i==v.size()-1) {
            fixedAngles.push_back(true);
          }
          else {
            fixedAngles.push_back(false);
          }
        }
        std::vector<real_type> curveParamV={Ks[j]};
        real_type* curveParam=curveParamV.data();
        
        std::string variant="AllInOneSlowly";
        std::string path=nameTest(deviceProperties.name, variant, "")+"/"+nExec+"/";
        std::cout << path << std::endl;
        std::string powerName=std::to_string(testI)+".log";
        std::string powerFile=path+powerName;

        system((std::string("mkdir -p ")+path).c_str());
        system((std::string("tegrastats --interval 50 --start --logfile ")+powerName).c_str());
        sleep(2);
        
        TimePerf tp, tp1;
        
        tp.start();
        DP::solveDPAllIn1<Dubins<double> >(v, discr, fixedAngles, curveParamV, false, jump);
        auto time1=tp.getTime();
        Run r1(nameTest(deviceProperties.name, variant).c_str(), discr, time1, testsNames[j], (nExec!="" ? powerFile : ""));
        r1.write(json_out);
        
        sleep(2);
        system((std::string("tegrastats --stop && mv ")+powerName+" "+powerFile).c_str());
        testI++;
        cout << "\tExample " << j+1 << std::setw(20) << std::setprecision(5) << time1 << "ms\t" << std::endl; //<< std::setw(20) << std::setprecision(5) <<  time2 << "ms\t" << std::setw(10) << (time2-time1) << "ms" << endl;
        json_out.close();
      //}
    //}
    //fstream json_out; json_out.open("tests.json", std::fstream::app);
    //json_out << "]}\n";
    //json_out.close();
  //}
#else
  for (int testID=0; testID<6; testID++){
    if (testID!=3){continue;}
    real_type dLen=exampleLenghts[testID];

    std::vector<bool> fixedAngles;
    for (uint i=0; i<Tests[testID].size(); i++){
      if (i==0 || i==Tests[testID].size()-1) {
        fixedAngles.push_back(true);
      }
      else {
        fixedAngles.push_back(false);
      }
    }
    std::vector<real_type> curveParamV={Ks[testID], 3};
    real_type* curveParam=curveParamV.data();
    
    for (auto DISCR :  discrs){
      //if (DISCR!=4){continue;}
      for (auto r : refins){
        //if (r!=1){continue;}
        //r=5;
        //std::cout << DISCR << " " << r << " ";
        TimePerf tp, tp1;
        tp.start();

        std::vector<Configuration2<double> >points=Tests[testID];
        DP::solveDP<Dubins<real_type> >(points, DISCR, fixedAngles, curveParamV, 1, true, r); 
        auto time1=tp.getTime();

        LEN_T Length=0.0;
        for (unsigned int idjijij=points.size()-1; idjijij>0; idjijij--){
          Dubins<real_type> c(points[idjijij-1], points[idjijij], Ks[testID]);
          Length+=c.l();
        }

        printf("%3d & %2d & ", DISCR, r);
        PrintScientific2D((Length-exampleLenghts[testID])*1000);
        printf(" & ");
        PrintScientific1D(time1);
        printf("&%.16f\\\\\n", Length);
        //cout << "Length: " << setprecision(30) << Length << " " << setprecision(20) << (ABS<real_type>(Length*1000.0, dLen*1000.0)) << endl;
        //cout << "Elapsed: " << std::setw(10) << time1 << "ms\t" << std::endl; // std::setw(10) << time2 << "ms\t" << std::setw(10) << (time2-time1) << "ms" << endl;
      }
    }
    printf("\n\n\n\n");
  }
#endif
  return 0;
}
