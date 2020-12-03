#include<iostream>
#include<fstream>
#include<assert.h>
#include<stdlib.h>

#define rj rapidjson

#include<rapidjson/document.h>
#include<rapidjson/istreamwrapper.h>

class GPU {
public:
  bool enabled;
  std::string additional_info;
  std::string perf_exec;
  int exec_t;

  GPU() : enabled(false), additional_info(""), perf_exec(""), exec_t(0) {}

  GPU(bool _enabled, std::string _additional_info, std::string _perf_exec, int _exec_t) : 
    enabled(_enabled), additional_info(_additional_info), perf_exec(_perf_exec), exec_t(_exec_t) {} 
};

class CPU {
public:
  bool enabled;
  std::string name;
  std::string perf_exec;
  int exec_t;

  CPU() : enabled(false), name(""), perf_exec(""), exec_t(0) {}

  CPU(bool _enabled, std::string _name, std::string _perf_exec, int _exec_t) : 
    enabled(_enabled), name(_name), perf_exec(_perf_exec), exec_t(_exec_t) {} 
};

int main(){
  std::ifstream ifs("include/settings.json");
  rj::IStreamWrapper isw(ifs);
  rj::Document d;
  d.ParseStream(isw);

  bool cmake=false, make=false;

  assert(d["cmake"].IsBool());
  cmake=d["cmake"].GetBool();
  assert(d["make"].IsBool());
  make=d["make"].GetBool();
  assert(cmake!=make);
  
  GPU g;
  CPU c;

  assert(d["GPU"]["enabled"].IsBool());
  g.enabled=d["GPU"]["enabled"].GetBool();

  assert(d["CPU"]["enabled"].IsBool());
  c.enabled=d["CPU"]["enabled"].GetBool();
  assert(c.enabled!=g.enabled);
  
  if (g.enabled){
    //Check values have correct types
    assert(d["GPU"]["additional_info"].IsString());
    assert(d["GPU"]["perf_exec"].IsString());
    assert(d["GPU"]["exec_t"].IsInt());
    //Save values 
    g.additional_info=d["GPU"]["additional_info"].GetString();
    g.perf_exec=d["GPU"]["perf_exec"].GetString();
    g.exec_t=d["GPU"]["exec_t"].GetInt();
    //Check values are correct
    assert(system(g.perf_exec)!=0)
    assert(g.exec_t>0)
  }
  if (c.enabled){
  
  }

  return 0;
}

