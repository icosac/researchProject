#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
#include<stdlib.h>
#include<unistd.h>

// #define DEBUG

#include<utils.cuh>
#include<dubins.cuh>
#include<dp.cuh>
#include<timeperf.hh>
// #include<constants.cuh>

#include<tests.hh>


void PrintScientific1D(double d){
  if (d == 0)
  {
    printf ("%*d", 6, 0);
    return;
  }

  int exponent  = (int)floor(log10( fabs(d)));  // This will round down the exponent
  double base   = d * pow(10, -1.0*exponent);

  printf("%1.1lfe%+01d", base, exponent);
}

void PrintScientific2D(double d){
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
std::vector<uint> discrs = {4, 16, 90, 360, 720};
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

  //std::cout << "CUDA" << std::endl;
  cudaFree(0);

  int devicesCount;
  cudaGetDeviceCount(&devicesCount);
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, 0);
  //printf("[%d] %s\n", 0, deviceProperties.name);

  if (argc==1){
    for (int testID=0; testID<6; testID++){
      if (testID!=2){continue;}
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
        if (DISCR>360){continue;}
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
          //std::cout << "Length: " << std::setprecision(30) << Length << " " << std::setprecision(20) << (ABS<real_type>(Length*1000.0, dLen*1000.0)) << endl;
          //std::cout << "Elapsed: " << std::setw(10) << time1 << "ms\t" << std::endl; // std::setw(10) << time2 << "ms\t" << std::setw(10) << (time2-time1) << "ms" << endl;
        }
      }
      printf("\n\n\n\n");
    }
  }

  else if (argc==9 || argc==10) {
    std::string testName=std::string(argv[1]);
    std::string nExec=std::string(argv[2]);
    uint testID=atoi(argv[3]);
    uint discr=atoi(argv[4]);
    uint rip=atoi(argv[5]);
    uint funcID=atoi(argv[6]);
    uint jump=0;
    if (funcID==2 && argc==10){ jump=atoi(argv[9]);}
    else if (funcID==2 && argc<10) {std::cerr << "Error, no number of jump passed to function" << std::endl; return 1; }
    bool guessAnglesVal=(atoi(argv[7])==1 ? true : false);
    uint threads=atoi(argv[8]);

    //std::cout << "testName: " << testName << std::endl;
    //std::cout << "nExec: " << nExec << std::endl;
    //std::cout << "testID: " << testID << std::endl;
    //std::cout << "discr: " << discr << std::endl;
    //std::cout << "rip: " << rip << std::endl;
    //std::cout << "funcID: " << funcID << std::endl;
    //std::cout << "jump: " << jump << std::endl;
    //std::cout << "guessAnglesVal: " << guessAnglesVal << std::endl;
    //std::cout << "threads: " << threads << std::endl;

    std::fstream json_out; json_out.open("testResults/tests.json", std::fstream::app);
    
    std::vector<bool> fixedAngles;
    vector<Configuration2<double> > v=Tests[testID];
    for (uint i=0; i<v.size(); i++){
      if (i==0 || i==v.size()-1) {
        fixedAngles.push_back(true);
      }
      else {
        fixedAngles.push_back(false);
      }
    }
    std::vector<real_type> curveParamV={Ks[testID]};
    if(jump!=0){ curveParamV.push_back(jump); }

    std::string variant=testsNames[testID]; std::replace(variant.begin(), variant.end(), ' ', '_');
    std::string path=testName+"/"+nExec+"/";
    //std::cout << path << std::endl;
    std::string powerName=variant+"_"+std::to_string(discr)+"_"+std::to_string(rip)+"_"+std::to_string(funcID)+"_"+std::to_string(guessAnglesVal ? 1 : 0)+"_"+std::to_string(threads)+"_"+std::to_string(jump)+".log";
    std::string powerFile=path+powerName;
    
    system((std::string("mkdir -p ")+path).c_str());
    system((std::string("tegrastats --interval 50 --start --logfile ")+powerName).c_str());
    //std::cout << (std::string("tegrastats --interval 50 --start --logfile ")+powerName).c_str() << std::endl;
    sleep(2);
    
    std::vector<Configuration2<real_type> > points=Tests[testID];

    TimePerf tp;
    tp.start();
    DP::solveDP<Dubins<real_type> >(points, discr, fixedAngles, curveParamV, funcID, guessAnglesVal, rip, threads); 
    auto time1=tp.getTime();

    LEN_T Length=0.0;
    for (unsigned int j=points.size()-1; j>0; j--){
      Dubins<real_type> c(points[j-1], points[j], Ks[testID]);
      Length+=c.l();
    }
    
    //std::cout << "Length: " << std::setprecision(30) << Length << " " << std::setprecision(20) << (ABS<real_type>(Length*1000.0, exampleLenghts[testID]*1000.0)) << endl;
    Run r1(testName, discr, time1, Length, ((Length-exampleLenghts[testID])*1000.0), testsNames[testID], rip, threads, funcID, jump, (guessAnglesVal ? "true" : "false"), (nExec!="" ? powerFile : ""));
    r1.write(json_out);
    json_out.close();
    
    sleep(2);
    system((std::string("tegrastats --stop && mv ")+powerName+" "+powerFile).c_str());
    //std::cout << ((std::string("tegrastats --stop && mv ")+powerName+" "+powerFile).c_str()) << std::endl;
    //std::cout << "\tExample " << std::setw(20) << std::setprecision(17) << time1 << "ms\t" << std::endl; //<< std::setw(20) << std::setprecision(5) <<  time2 << "ms\t" << std::setw(10) << (time2-time1) << "ms" << endl;
  }
  return 0;
}
