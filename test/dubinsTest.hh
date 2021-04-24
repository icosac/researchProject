//JUST TESTS DOING TESTS
#ifndef DUBINSTEST
#define DUBINSTEST

#if defined(BOOST)
#define BOOST_TEST_MODULE Dubins
#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>

//NOTE you can activate COUT by setting at compiling time DEBUG=ON to activate debug globally or DEBUG=TS to activate debug only for the tests. 

BOOST_AUTO_TEST_SUITE(SingleDubinsTest)
BOOST_AUTO_TEST_CASE(OneDubins){
  uint nerr=0,i=0;

  ifstream inputF("dubinsTestNew.txt"); 
  if(!inputF.is_open()){BOOST_ERROR("File not opened"); }

  real_type pl, ps1, ps2, ps3, pk1, pk2, pk3; 
  real_type el=0.0, es1=0.0, es2=0.0, es3=0.0, ek1=0.0, ek2=0.0, ek3=0.0; 
  int_type nl=0, ns1=0, ns2=0, ns3=0, nk1=0, nk2=0, nk3=0; 
  real_type x0, y0, th0, x1, y1, th1, kmax, l, s1, s2, s3, k1, k2, k3;
  while ( inputF >> kmax >> x0 >> y0 >> th0 >> x1 >> y1 >> th1 >>
          l >> s1 >> s2 >> s3 >> k1 >> k2 >> k3 >>
          pl >> ps1 >> ps2 >> ps3 >> pk1 >> pk2 >> pk3){
    bool berr=false;
    Configuration2<real_type>ci(x0, y0, th0);
    Configuration2<real_type>cf(x1, y1, th1);
    Dubins<real_type> d=solveDubins(ci, cf, kmax);
    if (!eq<real_type>(d.l(), l, pl))    { 
      berr=true; 
      COUT(" l: "  << std::setprecision(-1*(int)(log(pl)/log(10))+(int)(log(l)/log(10)))   << d.l()  << "-" << l  << "=" << ABS<real_type>(d.l(),l)   << ">" << pl  << endl)
      nl++;  el+=100.0*(ABS<real_type>(ABS<real_type>(d.l(),l), pl)/(ABS<real_type>(d.l(),l)));     
    }
    if (!eq<real_type>(d.k1(), k1, pk1)) { 
      berr=true; 
      COUT(" k1: " << std::setprecision(-1*(int)(log(pk1)/log(10))+(int)(log(k1)/log(10))) << d.k1() << "-" << k1 << "=" << ABS<real_type>(d.k1(),k1) << ">" << pk1 << endl)
      nk1++; ek1+=100.0*(ABS<real_type>(ABS<real_type>(d.k1(),k1), pk1)/(ABS<real_type>(d.k1(),k1)));
    }
    if (!eq<real_type>(d.k2(), k2, pk2)) { 
      berr=true; 
      COUT(" k2: " << std::setprecision(-1*(int)(log(pk2)/log(10))+(int)(log(k2)/log(10))) << d.k2() << "-" << k2 << "=" << ABS<real_type>(d.k2(),k2) << ">" << pk2 << endl)
      nk2++; ek2+=100.0*(ABS<real_type>(ABS<real_type>(d.k2(),k2), pk2)/(ABS<real_type>(d.k2(),k2)));
    }
    if (!eq<real_type>(d.k3(), k3, pk3)) { 
      berr=true; 
      COUT(" k3: " << std::setprecision(-1*(int)(log(pk3)/log(10))+(int)(log(k3)/log(10))) << d.k3() << "-" << k3 << "=" << ABS<real_type>(d.k3(),k3) << ">" << pk3 << endl)
      nk3++; ek3+=100.0*(ABS<real_type>(ABS<real_type>(d.k3(),k3), pk3)/(ABS<real_type>(d.k3(),k3)));
    }
    if (!eq<real_type>(d.s1(), s1, ps1)) { 
      berr=true; 
      COUT(" s1: " << std::setprecision(-1*(int)(log(ps1)/log(10))+(int)(log(s1)/log(10))) << d.s1() << "-" << s1 << "=" << ABS<real_type>(d.s1(),s1) << ">" << ps1 << endl)
      ns1++; es1+=100.0*(ABS<real_type>(ABS<real_type>(d.s1(),s1), ps1)/(ABS<real_type>(d.s1(),s1)));
    }
    if (!eq<real_type>(d.s2(), s2, ps2)) { 
      berr=true; 
      COUT(" s2: " << std::setprecision(-1*(int)(log(ps2)/log(10))+(int)(log(s2)/log(10))) << d.s2() << "-" << s2 << "=" << ABS<real_type>(d.s2(),s2) << ">" << ps2 << endl)
      ns2++; es2+=100.0*(ABS<real_type>(ABS<real_type>(d.s2(),s2), ps2)/(ABS<real_type>(d.s2(),s2)));
    }
    if (!eq<real_type>(d.s3(), s3, ps3)) { 
      berr=true; 
      COUT(" s3: " << std::setprecision(-1*(int)(log(ps3)/log(10))+(int)(log(s3)/log(10))) << d.s3() << "-" << s3 << "=" << ABS<real_type>(d.s3(),s3) << ">" << ps3 << endl)
      ns3++; es3+=100.0*(ABS<real_type>(ABS<real_type>(d.s3(),s3), ps3)/(ABS<real_type>(d.s3(),s3)));
    }
    if (berr){
      nerr++;
    }
    i+=1;
  } 
  inputF.close();

  if (nerr>0){
    cout << "Average errors over errors (%) for single Dubins: \n" <<
      "|     l     |     k1    |     k2    |     k3    |     s1    |     s2    |     s3    |\n" <<
      "|-----------------------------------------------------------------------------------|\n" <<
      std::setprecision(5) << std::scientific << 
      "|" << (nl>0 ? (el/nl) : 0.0) << 
      "|" << (nk1>0 ? (ek1/nk1) : 0.0) << 
      "|" << (nk2>0 ? (ek2/nk2) : 0.0) << 
      "|" << (nk3>0 ? (ek3/nk3) : 0.0) << 
      "|" << (ns1>0 ? (es1/ns1) : 0.0) << 
      "|" << (ns2>0 ? (es2/ns2) : 0.0) << 
      "|" << (ns3>0 ? (es3/ns3) : 0.0) << "|" << endl;
  }
  
  if(nerr/i>CONFIDENCE){
    BOOST_ERROR(boost::format("%1% tests failed over %2% total tests.\n") % nerr % i);
  }
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
    BOOST_ERROR(boost::format("Length of kaya1 %1$.12f does not match %2$.12f\n") % len % kaya1.res);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample2){
  LEN_T len=solve(kaya2.points, kaya2.params[0]);
  if(!eq(len, kaya2.res, EPSILON)){
    BOOST_ERROR(boost::format("Length of kaya2 %1$.12f does not match %2$.12f\n") % len % kaya2.res);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample3){
  LEN_T len=solve(kaya3.points, kaya3.params[0]);
  if(!eq(len, kaya3.res, EPSILON)){
    BOOST_ERROR(boost::format("Length of kaya3 %1$.12f does not match %2$.12f\n") % len % kaya3.res);
  }
}

BOOST_AUTO_TEST_CASE(KayaExample4){
  LEN_T len=solve(kaya4.points, kaya4.params[0]);
  if(!eq(len, kaya4.res, EPSILON)){
    BOOST_ERROR(boost::format("Length of kaya4 %1$.12f does not match %2$.12f\n") % len % kaya4.res);
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
    BOOST_ERROR(boost::format("Length of omega %1$.12f does not match %2$.12f\n") % len % omega.res);
  }
}

BOOST_AUTO_TEST_CASE(CircuitExample){
  LEN_T len=solve(circuit.points, circuit.params[0]);
  if(!eq(len, circuit.res, EPSILON)){
    BOOST_ERROR(boost::format("Length of circuit %1$.12f does not match %2$.12f\n") % len % circuit.res);
  }
}

BOOST_AUTO_TEST_SUITE_END()
#elif defined(GTEST)
#include <gtest/gtest.h>

#define TEST_COUT std::cout << "[          ] [ INFO ]"

TEST(DubinsTestName, OneDubins){//TODO Find bug for which ctest shows this test passed, but ./build/DubinsTest does not pass.
  real_type err;
  uint nerr=0,i=0;

  ifstream inputF("dubinsTestNew.txt"); 
  if(!inputF.is_open()){FAIL() << "File not opened"; }

  real_type pl, ps1, ps2, ps3, pk1, pk2, pk3; 
  real_type el=0.0, es1=0.0, es2=0.0, es3=0.0, ek1=0.0, ek2=0.0, ek3=0.0; 
  int_type nl=0, ns1=0, ns2=0, ns3=0, nk1=0, nk2=0, nk3=0; 
  real_type x0, y0, th0, x1, y1, th1, kmax, l, s1, s2, s3, k1, k2, k3;
  while ( inputF >> kmax >> x0 >> y0 >> th0 >> x1 >> y1 >> th1 >>
          l >> s1 >> s2 >> s3 >> k1 >> k2 >> k3 >>
          pl >> ps1 >> ps2 >> ps3 >> pk1 >> pk2 >> pk3){
    bool berr=false;
    Configuration2<real_type>ci(x0, y0, th0);
    Configuration2<real_type>cf(x1, y1, th1);
    Dubins<real_type> d=solveDubins(ci, cf, kmax);
    if (!eq<real_type>(d.l(), l, pl))    { 
      berr=true; 
      COUT(" l: "  << std::setprecision(-1*(int)(log(pl)/log(10))+(int)(log(l)/log(10)))   << d.l()  << "-" << l  << "=" << ABS<real_type>(d.l(),l)   << ">" << pl  << endl)
      nl++;  el+=100.0*(ABS<real_type>(ABS<real_type>(d.l(),l), pl)/(ABS<real_type>(d.l(),l)));     
    }
    if (!eq<real_type>(d.k1(), k1, pk1)) { 
      berr=true; 
      COUT(" k1: " << std::setprecision(-1*(int)(log(pk1)/log(10))+(int)(log(k1)/log(10))) << d.k1() << "-" << k1 << "=" << ABS<real_type>(d.k1(),k1) << ">" << pk1 << endl)
      nk1++; ek1+=100.0*(ABS<real_type>(ABS<real_type>(d.k1(),k1), pk1)/(ABS<real_type>(d.k1(),k1)));
    }
    if (!eq<real_type>(d.k2(), k2, pk2)) { 
      berr=true; 
      COUT(" k2: " << std::setprecision(-1*(int)(log(pk2)/log(10))+(int)(log(k2)/log(10))) << d.k2() << "-" << k2 << "=" << ABS<real_type>(d.k2(),k2) << ">" << pk2 << endl)
      nk2++; ek2+=100.0*(ABS<real_type>(ABS<real_type>(d.k2(),k2), pk2)/(ABS<real_type>(d.k2(),k2)));
    }
    if (!eq<real_type>(d.k3(), k3, pk3)) { 
      berr=true; 
      COUT(" k3: " << std::setprecision(-1*(int)(log(pk3)/log(10))+(int)(log(k3)/log(10))) << d.k3() << "-" << k3 << "=" << ABS<real_type>(d.k3(),k3) << ">" << pk3 << endl)
      nk3++; ek3+=100.0*(ABS<real_type>(ABS<real_type>(d.k3(),k3), pk3)/(ABS<real_type>(d.k3(),k3)));
    }
    if (!eq<real_type>(d.s1(), s1, ps1)) { 
      berr=true; 
      COUT(" s1: " << std::setprecision(-1*(int)(log(ps1)/log(10))+(int)(log(s1)/log(10))) << d.s1() << "-" << s1 << "=" << ABS<real_type>(d.s1(),s1) << ">" << ps1 << endl)
      ns1++; es1+=100.0*(ABS<real_type>(ABS<real_type>(d.s1(),s1), ps1)/(ABS<real_type>(d.s1(),s1)));
    }
    if (!eq<real_type>(d.s2(), s2, ps2)) { 
      berr=true; 
      COUT(" s2: " << std::setprecision(-1*(int)(log(ps2)/log(10))+(int)(log(s2)/log(10))) << d.s2() << "-" << s2 << "=" << ABS<real_type>(d.s2(),s2) << ">" << ps2 << endl)
      ns2++; es2+=100.0*(ABS<real_type>(ABS<real_type>(d.s2(),s2), ps2)/(ABS<real_type>(d.s2(),s2)));
    }
    if (!eq<real_type>(d.s3(), s3, ps3)) { 
      berr=true; 
      COUT(" s3: " << std::setprecision(-1*(int)(log(ps3)/log(10))+(int)(log(s3)/log(10))) << d.s3() << "-" << s3 << "=" << ABS<real_type>(d.s3(),s3) << ">" << ps3 << endl)
      ns3++; es3+=100.0*(ABS<real_type>(ABS<real_type>(d.s3(),s3), ps3)/(ABS<real_type>(d.s3(),s3)));
    }
    if (berr){
      nerr++;
    }
    i+=1;
  } 
  inputF.close();

  if (nerr>0){
    TEST_COUT << "Average errors over errors (%) for single Dubins: \n" <<
      "|     l     |     k1    |     k2    |     k3    |     s1    |     s2    |     s3    |\n" <<
      "|-----------------------------------------------------------------------------------|\n" <<
      std::setprecision(5) << std::scientific << 
      "|" << (nl>0 ? (el/nl) : 0.0) << 
      "|" << (nk1>0 ? (ek1/nk1) : 0.0) << 
      "|" << (nk2>0 ? (ek2/nk2) : 0.0) << 
      "|" << (nk3>0 ? (ek3/nk3) : 0.0) << 
      "|" << (ns1>0 ? (es1/ns1) : 0.0) << 
      "|" << (ns2>0 ? (es2/ns2) : 0.0) << 
      "|" << (ns3>0 ? (es3/ns3) : 0.0) << "|" << endl;
  }
  
  if(nerr/i>CONFIDENCE){
    FAIL() << nerr << " tests failed over " << i << " total tests." << endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  _  __                      _         _   _      _        _____                           _            // 
// | |/ /__ _ _   _  __ _     / \   _ __| |_(_) ___| | ___  | ____|_  ____ _ _ __ ___  _ __ | | ___  ___  //
// | ' // _` | | | |/ _` |   / _ \ | '__| __| |/ __| |/ _ \ |  _| \ \/ / _` | '_ ` _ \| '_ \| |/ _ \/ __| //
// | . \ (_| | |_| | (_| |  / ___ \| |  | |_| | (__| |  __/ | |___ >  < (_| | | | | | | |_) | |  __/\__ \ //
// |_|\_\__,_|\__, |\__,_| /_/   \_\_|   \__|_|\___|_|\___| |_____/_/\_\__,_|_| |_| |_| .__/|_|\___||___/ //
//            |___/                                                                   |_|                 // 
////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(DubinsTestName, KayaExample1){
  LEN_T len=solve(kaya1.points, kaya1.params[0]);
  EXPECT_NEAR(len, kaya1.res, EPSILON) << "Length " << len << " does not match " << kaya1.res << endl;
}

TEST(DubinsTestName, KayaExample2){
  LEN_T len=solve(kaya2.points, kaya2.params[0]);
  EXPECT_NEAR(len, kaya2.res, EPSILON) << "Length " << len << " does not match " << kaya2.res << endl; 
}

TEST(DubinsTestName, KayaExample3){
  LEN_T len=solve(kaya3.points, kaya3.params[0]);
  EXPECT_NEAR(len, kaya3.res, EPSILON) << "Length " << len << " does not match " << kaya3.res << endl; 
}

TEST(DubinsTestName, KayaExample4){
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

TEST(DubinsTestName, OmegaExample){
  LEN_T len=solve(omega.points, omega.params[0]);
  EXPECT_NEAR(len, omega.res, EPSILON) << "Length " << len << " does not match " << omega.res << endl; 
}

TEST(DubinsTestName, CircuitExample){
  LEN_T len=solve(circuit.points, circuit.params[0]);
  EXPECT_NEAR(len, circuit.res, EPSILON) << "Length " << len << " does not match " << circuit.res << endl; 
}
#endif

#endif //DUBINSTEST