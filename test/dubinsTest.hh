//JUST TESTS DOING TESTS
#ifndef DUBINSTEST
#define DUBINSTEST

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
  LEN_T len=solve(kaya1.points, kaya1.params[0]);
  if (!eq(len, kaya1.res, EPSILON)){ 
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % kaya1.res);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample2){
  LEN_T len=solve(kaya2.points, kaya2.params[0]);
  if(!eq(len, kaya2.res, EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % kaya2.res);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample3){
  LEN_T len=solve(kaya3.points, kaya3.params[0]);
  if(!eq(len, kaya3.res, EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % kaya3.res);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample4){
  LEN_T len=solve(kaya4.points, kaya4.params[0]);
  if(!eq(len, kaya4.res, EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % kaya4.res);
  }
}

///////////////////////////////////////////
//  _   _   _   _   ___   _____   _   _  //
// | | | | | \ | | |_ _| |_   _| | \ | | //
// | | | | |  \| |  | |    | |   |  \| | //
// | |_| | | |\  |  | |    | |   | |\  | //
//  \___/  |_| \_| |___|   |_|   |_| \_| //
//                                       //
///////////////////////////////////////////

BOOST_AUTO_TEST_CASE(OmegaExample){
  LEN_T len=solve(omega.points, omega.params[0]);
  if(!eq(len, omega.res, EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % omega.res);
  }
}

BOOST_AUTO_TEST_CASE(CircuitExample){
  LEN_T len=solve(circuit.points, circuit.params[0]);
  if(!eq(len, circuit.res, EPSILON)){
    BOOST_ERROR(boost::format("Length %1% does not match %2%\n") % len % circuit.res);
  }
}

BOOST_AUTO_TEST_SUITE_END()
#elif defined(GTEST)
#include <gtest/gtest.h>

#define TEST_COUT std::cout << "[          ] [ INFO ]"

TEST(DubinsTest, OneDubins){//TODO Find bug for which ctest shows this test passed, but ./build/DubinsTest does not pass.
  READ_FROM_FILE_DUBINS()
    Dubins<real_type> d(ci, cf, kmax);
    EXPECT_NEAR(d.l(), l, 1e-03);
    EXPECT_NEAR(d.s1(), s1, 1e-03);
    EXPECT_NEAR(d.s2(), s2, 1e-03);
    EXPECT_NEAR(d.s3(), s3, 1e-03);
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
  LEN_T len=solve(kaya1.points, kaya1.params[0]);
  EXPECT_NEAR(len, kaya1.res, EPSILON) << "Length " << len << " does not match " << kaya1.res << endl;
}

TEST(DubinsTest, KayaExample2){
  LEN_T len=solve(kaya2.points, kaya2.params[0]);
  EXPECT_NEAR(len, kaya2.res, EPSILON) << "Length " << len << " does not match " << kaya2.res << endl; 
}

TEST(DubinsTest, KayaExample3){
  LEN_T len=solve(kaya3.points, kaya3.params[0]);
  EXPECT_NEAR(len, kaya3.res, EPSILON) << "Length " << len << " does not match " << kaya3.res << endl; 
}

TEST(DubinsTest, KayaExample4){
  LEN_T len=solve(kaya4.points, kaya4.params[0]);
  EXPECT_NEAR(len, kaya4.res, EPSILON) << "Length " << len << " does not match " << kaya4.res << endl; 
}

///////////////////////////////////////////
//  _   _   _   _   ___   _____   _   _  //
// | | | | | \ | | |_ _| |_   _| | \ | | //
// | | | | |  \| |  | |    | |   |  \| | //
// | |_| | | |\  |  | |    | |   | |\  | //
//  \___/  |_| \_| |___|   |_|   |_| \_| //
//                                       //
///////////////////////////////////////////

TEST(DubinsTest, OmegaExample){
  LEN_T len=solve(omega.points, omega.params[0]);
  EXPECT_NEAR(len, omega.res, EPSILON) << "Length " << len << " does not match " << omega.res << endl; 
}

TEST(DubinsTest, CircuitExample){
  LEN_T len=solve(circuit.points, circuit.params[0]);
  EXPECT_NEAR(len, circuit.res, EPSILON) << "Length " << len << " does not match " << circuit.res << endl; 
}
#endif

#endif //DUBINSTEST