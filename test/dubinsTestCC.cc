#include<iostream>
#include<sstream>
#include<fstream>

#include<dp.hh>
#include<tests.hh>

using namespace std;

#define TESTONCUDA 0
#define DubinsTestName DubinsTestCPU
#include"dubinsTestUtils.hh"
#include"dubinsTest.hh" //PLEASE add tests here, this code is as device independent as possible
