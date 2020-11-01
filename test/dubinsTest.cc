#include<iostream>
#include<sstream>

#include<dp.hh>
#include <utils.hh>

using namespace std;

#define DISCR 720
#define EPSILON 0.0001

inline const char* toCString(std::stringstream msg){
  return msg.str().c_str();
}

const std::vector<LEN_T> KayaLengths={3.415578858075, 6.2780346, 11.916212654286, 7.467562181965};

LEN_T solve(std::vector<Configuration2<double> > kaya){
  std::vector<bool> fixedAngles;
  for (uint i=0; i<kaya.size(); i++){
    if (i==0 || i==kaya.size()-1) {
      fixedAngles.push_back(true);
    }
    else {
      fixedAngles.push_back(false);
    }
  }
  std::vector<Angle> bestA=DP::solveDP<Dubins<double> >(kaya, DISCR, fixedAngles);
  
  LEN_T Length=0.0;
  for (uint i=0; i<bestA.size()-1; i++){
    kaya[i].th(bestA[i]);
    kaya[i+1].th(bestA[i+1]);
    Dubins<double> c(kaya[i], kaya[i+1], KMAX);
    Length+=c.l();
  }

  return Length;
}

#if defined(BOOST)
#define BOOST_TEST_MODULE Dubins
#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>

BOOST_AUTO_TEST_SUITE(DubinsTest)
BOOST_AUTO_TEST_CASE(KayaExample1){
  vector<Configuration2<double> > kaya1={
          Configuration2<double> (0, 0, -M_PI/3.0),
          Configuration2<double> (-0.1, 0.3, ANGLE::INVALID),
          Configuration2<double> (0.2, 0.8, ANGLE::INVALID),
          Configuration2<double> (1, 1, -M_PI/6.0)
  };
  LEN_T len=solve(kaya1);
  if (!eq(len, KayaLengths[0], EPSILON)){ 
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % KayaLengths[0]);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample2){
  vector<Configuration2<double> > kaya2={
          Configuration2<double> (0, 0, -M_PI/3.0),
          Configuration2<double> (-0.1, 0.3, ANGLE::INVALID),
          Configuration2<double> (0.2, 0.8, ANGLE::INVALID),
          Configuration2<double> (1, 1, ANGLE::INVALID),
          Configuration2<double> (0.5, 0.5, ANGLE::INVALID),
          Configuration2<double> (0.5, 0, -M_PI/6.0)
  };
  LEN_T len=solve(kaya2);
  if(!eq(len, KayaLengths[1], EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % KayaLengths[1]);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample3){
  vector<Configuration2<double> > kaya3={
         Configuration2<double>(0.5, 1.2, 5.0*M_PI/6.0),
         Configuration2<double>(0, 0.8, ANGLE::INVALID),
         Configuration2<double>(0, 0.4, ANGLE::INVALID),
         Configuration2<double>(0.1, 0, ANGLE::INVALID),
         Configuration2<double>(0.4, 0.2, ANGLE::INVALID),
         Configuration2<double>(0.5, 0.5, ANGLE::INVALID),
         Configuration2<double>(0.6, 1, ANGLE::INVALID),
         Configuration2<double>(1, 0.8, ANGLE::INVALID),
         Configuration2<double>(1, 0, ANGLE::INVALID),
         Configuration2<double>(1.4, 0.2, ANGLE::INVALID),
         Configuration2<double>(1.2, 1, ANGLE::INVALID),
         Configuration2<double>(1.5, 1.2, ANGLE::INVALID),
         Configuration2<double>(2, 1.5, ANGLE::INVALID),
         Configuration2<double>(1.5, 0.8, ANGLE::INVALID),
         Configuration2<double>(1.5, 0, ANGLE::INVALID),
         Configuration2<double>(1.7, 0.6, ANGLE::INVALID),
         Configuration2<double>(1.9, 1, ANGLE::INVALID),
         Configuration2<double>(2, 0.5, ANGLE::INVALID),
         Configuration2<double>(1.9, 0, ANGLE::INVALID),
         Configuration2<double>(2.5, 0.6, 0),
  };
  LEN_T len=solve(kaya3);
  if(!eq(len, KayaLengths[2], EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % KayaLengths[2]);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample4){
  vector<Configuration2<double> > kaya4={
         Configuration2<double>(0.5, 1.2, 5*M_PI/6.0),
         Configuration2<double>(0.0, 0.5, ANGLE::INVALID),
         Configuration2<double>(0.5, 0.5, ANGLE::INVALID),
         Configuration2<double>(1.0, 0.5, ANGLE::INVALID),
         Configuration2<double>(1.5, 0.5, ANGLE::INVALID),
         Configuration2<double>(2.0, 0.5, ANGLE::INVALID),
         Configuration2<double>(2.0, 0.0, ANGLE::INVALID),
         Configuration2<double>(1.5, 0.0, ANGLE::INVALID),
         Configuration2<double>(1.0, 0.0, ANGLE::INVALID),
         Configuration2<double>(0.5, 0.0, ANGLE::INVALID),
         Configuration2<double>(0.0, 0.0, ANGLE::INVALID),
         Configuration2<double>(0.0, -0.5, 0)
  };
  LEN_T len=solve(kaya4);
  if(!eq(len, KayaLengths[3], EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % KayaLengths[3]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
#elif defined(GTEST)
#include <gtest/gtest.h>

#define TEST_COUT std::cerr << "[          ] [ INFO ]"



///////////////////////////////////////////////////////////////////////////////////////////////////////////
//  _  __                      _         _   _      _        _____                           _           // 
// | |/ /__ _ _   _  __ _     / \   _ __| |_(_) ___| | ___  | ____|_  ____ _ _ __ ___  _ __ | | ___  ___ //
// | ' // _` | | | |/ _` |   / _ \ | '__| __| |/ __| |/ _ \ |  _| \ \/ / _` | '_ ` _ \| '_ \| |/ _ \/ __|//
// | . \ (_| | |_| | (_| |  / ___ \| |  | |_| | (__| |  __/ | |___ >  < (_| | | | | | | |_) | |  __/\__ \//
// |_|\_\__,_|\__, |\__,_| /_/   \_\_|   \__|_|\___|_|\___| |_____/_/\_\__,_|_| |_| |_| .__/|_|\___||___///
//            |___/                                                                   |_|                // 
///////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(DubinsTest, KayaExample1){
  vector<Configuration2<double> > kaya1={
          Configuration2<double> (0, 0, -M_PI/3.0),
          Configuration2<double> (-0.1, 0.3, ANGLE::INVALID),
          Configuration2<double> (0.2, 0.8, ANGLE::INVALID),
          Configuration2<double> (1, 1, -M_PI/6.0)
  };
  LEN_T len=solve(kaya1);
  EXPECT_NEAR(len, KayaLengths[0], EPSILON) << "Length " << len << " does not match " << KayaLengths[0] << endl;
}

TEST(DubinsTest, KayaExample2){
  vector<Configuration2<double> > kaya2={
          Configuration2<double> (0, 0, -M_PI/3.0),
          Configuration2<double> (-0.1, 0.3, ANGLE::INVALID),
          Configuration2<double> (0.2, 0.8, ANGLE::INVALID),
          Configuration2<double> (1, 1, ANGLE::INVALID),
          Configuration2<double> (0.5, 0.5, ANGLE::INVALID),
          Configuration2<double> (0.5, 0, -M_PI/6.0)
  };
  LEN_T len=solve(kaya2);
  EXPECT_NEAR(len, KayaLengths[1], EPSILON) << "Length " << len << " does not match " << KayaLengths[1] << endl; 
}

TEST(DubinsTest, KayaExample3){
  vector<Configuration2<double> > kaya3={
         Configuration2<double>(0.5, 1.2, 5.0*M_PI/6.0),
         Configuration2<double>(0, 0.8, ANGLE::INVALID),
         Configuration2<double>(0, 0.4, ANGLE::INVALID),
         Configuration2<double>(0.1, 0, ANGLE::INVALID),
         Configuration2<double>(0.4, 0.2, ANGLE::INVALID),
         Configuration2<double>(0.5, 0.5, ANGLE::INVALID),
         Configuration2<double>(0.6, 1, ANGLE::INVALID),
         Configuration2<double>(1, 0.8, ANGLE::INVALID),
         Configuration2<double>(1, 0, ANGLE::INVALID),
         Configuration2<double>(1.4, 0.2, ANGLE::INVALID),
         Configuration2<double>(1.2, 1, ANGLE::INVALID),
         Configuration2<double>(1.5, 1.2, ANGLE::INVALID),
         Configuration2<double>(2, 1.5, ANGLE::INVALID),
         Configuration2<double>(1.5, 0.8, ANGLE::INVALID),
         Configuration2<double>(1.5, 0, ANGLE::INVALID),
         Configuration2<double>(1.7, 0.6, ANGLE::INVALID),
         Configuration2<double>(1.9, 1, ANGLE::INVALID),
         Configuration2<double>(2, 0.5, ANGLE::INVALID),
         Configuration2<double>(1.9, 0, ANGLE::INVALID),
         Configuration2<double>(2.5, 0.6, 0),
  };
  LEN_T len=solve(kaya3);
  EXPECT_NEAR(len, KayaLengths[2], EPSILON) << "Length " << len << " does not match " << KayaLengths[2] << endl; 
}

TEST(DubinsTest, KayaExample4){
  vector<Configuration2<double> > kaya4={
         Configuration2<double>(0.5, 1.2, 5*M_PI/6.0),
         Configuration2<double>(0.0, 0.5, ANGLE::INVALID),
         Configuration2<double>(0.5, 0.5, ANGLE::INVALID),
         Configuration2<double>(1.0, 0.5, ANGLE::INVALID),
         Configuration2<double>(1.5, 0.5, ANGLE::INVALID),
         Configuration2<double>(2.0, 0.5, ANGLE::INVALID),
         Configuration2<double>(2.0, 0.0, ANGLE::INVALID),
         Configuration2<double>(1.5, 0.0, ANGLE::INVALID),
         Configuration2<double>(1.0, 0.0, ANGLE::INVALID),
         Configuration2<double>(0.5, 0.0, ANGLE::INVALID),
         Configuration2<double>(0.0, 0.0, ANGLE::INVALID),
         Configuration2<double>(0.0, -0.5, 0)
  };
  LEN_T len=solve(kaya4);
  EXPECT_NEAR(len, KayaLengths[3], EPSILON) << "Length " << len << " does not match " << KayaLengths[3] << endl; 
}

#endif
