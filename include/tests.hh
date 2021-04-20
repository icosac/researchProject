#pragma once

#include<iomanip>
#include<vector>

#include<testsPoints.hh>

class Run{
public:
  std::string name, test_name, file_name, guessInitialAngles;
  double time, length, err, initTime, endTime;
  uint discr, threads, functionType, jump, rip;

  Run ( std::string _name, uint _discr, double _time, double _length, double _err,
        std::string _test_name, uint _rip, uint _threads, uint _functionType, 
        uint _jump, std::string  _guessInitialAngles, std::string _file_name="", 
        double _initTime=0, double _endTime=0) 
      : name(_name), discr(_discr), time(_time), length(_length), err(_err), test_name(_test_name), rip(_rip), 
        threads(_threads), functionType(_functionType), jump(_jump), guessInitialAngles(_guessInitialAngles), 
        file_name(_file_name), initTime(_initTime), endTime(_endTime)
  {}

  void write(std::fstream& out){
    out << 
      "{\"name\" : \"" << name << 
      "\", \"test_name\" : \"" << test_name <<  "\"" <<
      ", \"discr\" : " << discr << 
      ", \"time\" : " << time << 
      ", \"length\" : " << std::setprecision(20) << length << 
      ", \"err\" : " << std::setprecision(1) << std::scientific << err << 
      ", \"refinements\" : " << rip <<
      ", \"threads\" : " << threads <<
      ", \"functionType\" : " << functionType <<
      ", \"jump\" : " << jump <<
      ", \"guessInitialAngles\" : \"" << guessInitialAngles <<  "\"" << 
      (file_name=="" ? "" : ", \"power_file\" : \""+file_name+"\"") << 
      ", \"initTime\" : " << std::setprecision(20) << initTime << 
      ", \"endTime\" : " << std::setprecision(20) << endTime << 
      "},\n";
  }
};

class Test {
public:
  std::string name;
  std::vector<double> params;
  double res;
  std::vector<Configuration2<double> >points;
  
  Test( std::string _name, 
        std::vector<double> _params, 
        LEN_T _res,
        std::vector<Configuration2<double> > _points):
    name(_name), params(_params), res(_res), points(_points)
  {}  

  operator std::vector<Configuration2<double> >(){ return this->points; }
};

std::vector<K_T> testKs = {
  3.0, // Kaya 1 
  3.0, // Kaya 2
  5.0, // Kaya 3
  3.0, // Kaya 4
  3.0, // Omega
  3.0  // SPA
};

std::vector<LEN_T> testLengths={
        3.41557885807514871601142658619, // Kaya 1 
        6.27803455030931356617429628386, // Kaya 2 
        11.9162126542854860389297755319, // Kaya 3 
        7.46756219733842652175326293218, // Kaya 4 
        41.0725016438839318766440555919, // Omega
        6988.66098639942993031581863761  // SPA
      };

Test kaya1=Test(
  "kaya1",
  std::vector<double>(1, testKs[0]),
  testLengths[0],
  kaya1Points
);

Test kaya2=Test(
  "kaya2",
  std::vector<double>(1, testKs[1]),
  testLengths[1],
  kaya2Points
);

Test kaya3=Test(
  "kaya3",
  std::vector<double>(1, testKs[2]),
  testLengths[2],
  kaya3Points
);

Test kaya4=Test(
  "kaya4",
  std::vector<double>(1, testKs[3]),
  testLengths[3],
  kaya4Points
);

Test omega=Test(
  "omega",
  std::vector<double>(1, testKs[4]),
  testLengths[4],
  omegaPoints
);

Test circuit=Test(
  "circuit",
  std::vector<double>(1, testKs[5]),
  testLengths[5],
  circuitPoints
);



