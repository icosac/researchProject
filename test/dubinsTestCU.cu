#include<iostream>
#include<sstream>
#include<fstream>

#include<dp.cuh>
#include<tests.hh>

using namespace std;

#define DISCR 720
#define EPSILON 1e-4

inline const char* toCString(std::stringstream msg){
  return msg.str().c_str();
}

const std::vector<LEN_T> KayaLengths={3.415578858075, 6.2780346, 11.916212654286, 7.467562181965};

LEN_T solve(std::vector<Configuration2<real_type> > kaya, real_type kmax, short dp_func=1){
  //Compute fixedAngles, which are simply the first and the last one;
  std::vector<bool> fixedAngles;
  for (uint i=0; i<kaya.size(); i++){
    if (i==0 || i==kaya.size()-1) {
      fixedAngles.push_back(true);
    }
    else {
      fixedAngles.push_back(false);
    }
  }
  //Write parameters
  std::vector<real_type> params={kmax};
  if (dp_func==2) params.push_back(3.0);

  //Solve DP problem
  DP::solveDP<Dubins<real_type> >(kaya, DISCR, fixedAngles, params, dp_func, true); //Using initial angle guess for average better results.

  //Compute total length
  LEN_T Length=0.0;
  for (uint i=0; i<kaya.size()-1; i++){
    Dubins<real_type> c(kaya[i], kaya[i+1], kmax);
    Length+=c.l();
  }

  return Length;
}

#define READ_FROM_FILE_DUBINS()                                                                            \
  ifstream input("test/dubinsTest.txt");                                                                   \
    real_type x0, y0, th0, x1, y1, th1, kmax, l, s1, s2, s3, k1, k2, k3;                                   \
    int i=0;                                                                                               \
    while (input >> kmax >> x0 >> y0 >> th0 >> x1 >> y1 >> th1 >> l >> s1 >> s2 >> s3 >> k1 >> k2 >> k3){  \
      i++;                                                                                                 \
      Configuration2<real_type>ci(x0, y0, th0);                                                            \
      Configuration2<real_type>cf(x1, y1, th1);                                                            

#define CLOSE_FILE_DUBINS() } input.close();


#if defined(BOOST)
#define BOOST_TEST_MODULE Dubins
#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>

BOOST_AUTO_TEST_SUITE(SingleDubinsTest)
BOOST_AUTO_TEST_CASE(OneDubins){
  READ_FROM_FILE_DUBINS()
    Dubins<real_type> d(ci, cf, kmax);
    if (!eq(d.l(), l, 1e-03)){ BOOST_ERROR(boost::format("Length l %1% does not match %2%\n") % d.l() % l); }
    if (!eq(d.s1(), s1, 1e-03)){ BOOST_ERROR(boost::format("Length s1 %1% does not match %2%\n") % d.s1() % s1); }
    if (!eq(d.s2(), s2, 1e-03)){ BOOST_ERROR(boost::format("Length s2 %1% does not match %2%\n") % d.s2() % s2); }
    if (!eq(d.s3(), s3, 1e-03)){ BOOST_ERROR(boost::format("Length s3 %1% does not match %2%\n") % d.s3() % s3); }
    if (!eq(d.k1(), k1, EPSILON)){ BOOST_ERROR(boost::format("Curvature k1 %1% does not match %2%\n") % d.k1() % k1); }
    if (!eq(d.k2(), k2, EPSILON)){ BOOST_ERROR(boost::format("Curvature k2 %1% does not match %2%\n") % d.k2() % k2); }
    if (!eq(d.k3(), k3, EPSILON)){ BOOST_ERROR(boost::format("Curvature k3 %1% does not match %2%\n") % d.k3() % k3); }
  CLOSE_FILE_DUBINS()
}
BOOST_AUTO_TEST_SUITE_END()

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  _  __                      _         _   _      _        _____                           _            // 
// | |/ /__ _ _   _  __ _     / \   _ __| |_(_) ___| | ___  | ____|_  ____ _ _ __ ___  _ __ | | ___  ___  //
// | ' // _` | | | |/ _` |   / _ \ | '__| __| |/ __| |/ _ \ |  _| \ \/ / _` | '_ ` _ \| '_ \| |/ _ \/ __| //
// | . \ (_| | |_| | (_| |  / ___ \| |  | |_| | (__| |  __/ | |___ >  < (_| | | | | | | |_) | |  __/\__ \ //
// |_|\_\__,_|\__, |\__,_| /_/   \_\_|   \__|_|\___|_|\___| |_____/_/\_\__,_|_| |_| |_| .__/|_|\___||___/ //
//            |___/                                                                   |_|                 // 
////////////////////////////////////////////////////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_SUITE(MultiDubinsTest)
BOOST_AUTO_TEST_CASE(KayaExample1){
  LEN_T len=solve(kaya1, 3.0);
  if (!eq(len, KayaLengths[0], EPSILON)){ 
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % KayaLengths[0]);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample2){
  LEN_T len=solve(kaya2, 3.0);
  if(!eq(len, KayaLengths[1], EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % KayaLengths[1]);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample3){
  LEN_T len=solve(kaya3, 5.0);
  if(!eq(len, KayaLengths[2], EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % KayaLengths[2]);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample4){
  LEN_T len=solve(kaya4, 3.0);
  if(!eq(len, KayaLengths[3], EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % KayaLengths[3]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
#elif defined(GTEST)
#include <gtest/gtest.h>

#define TEST_COUT std::cout << "[          ] [ INFO ]"

TEST(DubinsTest, OneDubins){//TODO Find bug for which ctest shows this test passed, but ./build/DubinsTest does not pass.
  READ_FROM_FILE_DUBINS()
    Dubins<real_type> d(ci, cf, kmax);
    EXPECT_NEAR(d.l(), l, EPSILON);
    EXPECT_NEAR(d.s1(), s1, EPSILON);
    EXPECT_NEAR(d.s2(), s2, EPSILON);
    EXPECT_NEAR(d.s3(), s3, EPSILON);
    EXPECT_NEAR(d.k1(), k1, EPSILON);
    EXPECT_NEAR(d.k2(), k2, EPSILON);
    EXPECT_NEAR(d.k3(), k3, EPSILON);
  CLOSE_FILE_DUBINS()
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  _  __                      _         _   _      _        _____                           _            // 
// | |/ /__ _ _   _  __ _     / \   _ __| |_(_) ___| | ___  | ____|_  ____ _ _ __ ___  _ __ | | ___  ___  //
// | ' // _` | | | |/ _` |   / _ \ | '__| __| |/ __| |/ _ \ |  _| \ \/ / _` | '_ ` _ \| '_ \| |/ _ \/ __| //
// | . \ (_| | |_| | (_| |  / ___ \| |  | |_| | (__| |  __/ | |___ >  < (_| | | | | | | |_) | |  __/\__ \ //
// |_|\_\__,_|\__, |\__,_| /_/   \_\_|   \__|_|\___|_|\___| |_____/_/\_\__,_|_| |_| |_| .__/|_|\___||___/ //
//            |___/                                                                   |_|                 // 
////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(DubinsTest, KayaExample1){
  LEN_T len=solve(kaya1, 3.0);
  EXPECT_NEAR(len, KayaLengths[0], EPSILON) << "Length " << len << " does not match " << KayaLengths[0] << endl;
}

TEST(DubinsTest, KayaExample2){
  LEN_T len=solve(kaya2, 3.0);
  EXPECT_NEAR(len, KayaLengths[1], EPSILON) << "Length " << len << " does not match " << KayaLengths[1] << endl; 
}

TEST(DubinsTest, KayaExample3){
  LEN_T len=solve(kaya3, 5.0);
  EXPECT_NEAR(len, KayaLengths[2], EPSILON) << "Length " << len << " does not match " << KayaLengths[2] << endl; 
}

TEST(DubinsTest, KayaExample4){
  LEN_T len=solve(kaya4, 3.0);
  EXPECT_NEAR(len, KayaLengths[3], EPSILON) << "Length " << len << " does not match " << KayaLengths[3] << endl; 
}

#endif

