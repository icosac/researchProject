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
std::vector<uint> discrs = {4, 16, 90, 360};
//std::vector<uint> discrs = {4, 16, 90, 120, 360, 720, 1440};
std::vector<uint> refins = {1, 2, 4, 8, 16};
std::vector<LEN_T> exampleLenghts={3.41557885807514871601142658619, 6.27803455030931356617429628386, 11.9162126542854860389297755319, 7.46756219733842652175326293218, 41.0725016438839318766440555919, 6988.66098639942993031581863761}; //the last length is SPA

std::string nameTest(std::string name, std::string add="", std::string conc=" "){
  if (add==""){
    return name;
  }
  else{
    return name+conc+add;
  }
}

__global__ void dubinsL(Configuration2<double> c0, Configuration2<double> c1, double k, double* L){
  Dubins<double> dubins(c0, c1, k);
  L[0]+=dubins.l();
  //printf("GPU Length: %.16f\n", dubins.l());
}

int main (int argc, char* argv[]){

  //std::cout << "CUDA" << std::endl;
  cudaFree(0);

  int devicesCount;
  cudaGetDeviceCount(&devicesCount);
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, 0);
  //printf("[%d] %s\n", 0, deviceProperties.name);
 
/*
  double CPU_C=-0.000000018581287397623214019404;
  double CPU_S=0.000000037114354256573278689757;
  double GPU_C=-0.000000018581287397623214019404;
  double GPU_S=0.000000037114354478617883614788;
  printf("err C: %.16f\n", (CPU_C-GPU_C));
  printf("err S: %.16f\n", (CPU_S-GPU_S));

  tryAtan2<<<1,1>>>(GPU_C, GPU_S);
  cudaDeviceSynchronize();
  printf("CPU 2: %.16f\n", atan2(CPU_C, CPU_S));
  printf("CPU 1: %.16f\n", atan(CPU_C/CPU_S));
  //printf("CPU v: %.16f\n", M_PI-atan(0.000000037114354256573278689757/-0.000000018581287397623214019404));

  std::cout << "==================" << std::endl;

  tryAtan2<<<1,1>>>(CPU_C, CPU_S);
  cudaDeviceSynchronize();
  printf("CPU 2 using GPU: %.16f\n", atan2(GPU_C, GPU_S));
  printf("CPU 1 using GPU: %.16f\n", atan(GPU_C/GPU_S));


  std::cout << "==================" << std::endl;

  Configuration2<double> c0(2.0, 0.5, -0.72273426348170455);
  Configuration2<double> c1(2.0, 0.0, -2.4188583653330378);
  std::cout << std::setw(20) << std::setprecision(17);
  Dubins<double> dubins(c0, c1, 3.0);
  std::cout << "CPU length: " << std::setw(20) << std::setprecision(17) << dubins.l() << std::endl;
  dubinsL<<<1,1>>>(c0.x(), c0.y(), c0.th(), c1.x(), c1.y(), c1.th(), 3.0);
  cudaDeviceSynchronize();
  //return 0;

  std::vector<double> x={ 0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2, 1.5, 1, 0.5, 0., 0. };
  std::vector<double> y={ 1.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.0, 0, 0.0, 0., -0.5 };
  std::vector<double> th={2.6179938779914944, 5.5181700068343345, 0.27550703827297418, 6.2600960773072822, 0.19802151651795322, 5.5604509844293766, 3.8643269723697231, 2.9435898904162778, 3.1477811860951026, 3.2289796515706222, 2.580176702570077, 0};
  double totL=0.0;
  for (int i=x.size()-2; i>=0; i--){
    std::cout << "i: " << i << std::endl;
    Configuration2<double> c0(x[i], y[i], th[i]);
    Configuration2<double> c1(x[i+1], y[i+1], th[i+1]);
    double *L; cudaMallocManaged(&L, sizeof(double));
    DP::dubinsWrapper<<<1,1>>>(c0, c1, 3.0, L);
    cudaDeviceSynchronize();
    totL+=L[0];
    std::cout << "L: " << std::setw(20) << std::setprecision(17) << L[0] << std::endl;
    cudaFree(L);
  }
  std::cout << "totL: " << std::setw(20) << std::setprecision(17) << totL << std::endl;
  return 0;
  */
  if (argc==1){
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
        if (DISCR!=360){continue;}
        for (auto r : refins){
          if (r!=16){continue;}
          //r=5;
          //std::cout << DISCR << " " << r << " ";
          TimePerf tp, tp1;
          tp.start();

          std::vector<Configuration2<double> >points=Tests[testID];
          DP::solveDP<Dubins<real_type> >(points, DISCR, fixedAngles, curveParamV, 1, true, r); 
          auto time1=tp.getTime();
          LEN_T Length=0.0;
          LEN_T *Length1; cudaMallocManaged(&Length1, sizeof(LEN_T));
          for (unsigned int idjijij=points.size()-1; idjijij>0; idjijij--){
            dubinsL<<<1,1>>>(points[idjijij-1], points[idjijij], Ks[testID], Length1);
            cudaDeviceSynchronize();

            Dubins<real_type> c(points[idjijij-1], points[idjijij], Ks[testID]);
            Length+=c.l();
          }

          printf("%3d & %2d & ", DISCR, r);
          PrintScientific2D((Length-exampleLenghts[testID])*1000.0);
          printf(" & ");
          PrintScientific2D((Length1[0]-Length)*1000.0);
          printf(" & ");
          PrintScientific2D((Length1[0]-exampleLenghts[testID])*1000.0);
          printf(" & ");
          PrintScientific1D(time1);
          printf("&%.16f", Length);
          printf("&%.16f\\\\\n", Length1[0]);

          cudaFree(Length1);
          //std::cout << "Length: " << std::setprecision(30) << Length << " " << std::setprecision(20) << (ABS<real_type>(Length*1000.0, dLen*1000.0)) << endl;
          //std::cout << "Elapsed: " << std::setw(10) << time1 << "ms\t" << std::endl; // std::setw(10) << time2 << "ms\t" << std::setw(10) << (time2-time1) << "ms" << endl;
        }
      }
      printf("\n\n\n\n");
    }
  }

  else if (argc>=9) {
    std::string testName=std::string(argv[1]);
    std::string nExec=std::string(argv[2]);
    uint testID=atoi(argv[3]);
    uint discr=atoi(argv[4]);
    uint rip=atoi(argv[5]);
    uint funcID=atoi(argv[6]);
    uint jump=0;
    if (funcID==2 && argc>9){ jump=atoi(argv[9]);}
    else if (funcID==2 && argc<10) {std::cerr << "Error, no number of jump passed to function" << std::endl; return 1; }
    bool guessAnglesVal=(atoi(argv[7])==1 ? true : false);
    uint threads=atoi(argv[8]);

    double initTime=-1.0;
    double endTime=0.0;

    if (argc==11){
      initTime=atof(argv[10]);
    }

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
    
    //system((std::string("mkdir -p ")+path).c_str());
    //system((std::string("tegrastats --interval 50 --start --logfile ")+powerName).c_str());
    //std::cout << (std::string("tegrastats --interval 50 --start --logfile ")+powerName).c_str() << std::endl;
    sleep(2);
    
    std::vector<Configuration2<real_type> > points=Tests[testID];

    TimePerf tp;
    tp.start();
    DP::solveDP<Dubins<real_type> >(points, discr, fixedAngles, curveParamV, funcID, guessAnglesVal, rip, threads); 
    auto time1=tp.getTime();

    if (initTime!=-1.0){
      endTime=initTime+time1;
    }

    LEN_T Length=0.0;
    for (unsigned int j=points.size()-1; j>0; j--){
      Dubins<real_type> c(points[j-1], points[j], Ks[testID]);
      Length+=c.l();
    }
    
    //std::cout << "Length: " << std::setprecision(30) << Length << " " << std::setprecision(20) << (ABS<real_type>(Length*1000.0, exampleLenghts[testID]*1000.0)) << endl;
    Run r1(testName, discr, time1, Length, ((Length-exampleLenghts[testID])*1000.0), testsNames[testID], rip, threads, funcID, jump, (guessAnglesVal ? "true" : "false"), (nExec!="" ? powerFile : ""), initTime, endTime);
    r1.write(json_out);
    json_out.close();
    
    sleep(2);
    //system((std::string("tegrastats --stop && mv ")+powerName+" "+powerFile).c_str());
    //std::cout << ((std::string("tegrastats --stop && mv ")+powerName+" "+powerFile).c_str()) << std::endl;
    //std::cout << "\tExample " << std::setw(20) << std::setprecision(17) << time1 << "ms\t" << std::endl; //<< std::setw(20) << std::setprecision(5) <<  time2 << "ms\t" << std::setw(10) << (time2-time1) << "ms" << endl;
  }
  return 0;
}
