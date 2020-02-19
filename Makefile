MKDIR=mkdir -p
OS=$(shell uname)
CLR=clear && clear && clear

CC=g++
CCFLAGS=-std=c++11
NV=nvcc
NVFLAGS=-std=c++11 -arch=sm_35 -rdc=true --default-stream per-thread

AR=ar rcs

CCSRC=$(wildcard src/*.cc)
NVSRC=$(wildcard src/*.cu)
CCOBJ=$(subst src,src/obj/cc/,$(patsubst %.cc,%.o, $(SRC)))
NVOBJ=$(subst src,src/obj/cu/,$(patsubst %.cu,%.o, $(SRC)))

TESTCCSRC=$(wildcard test/*.cc)
TESTNVSRC=$(wildcard test/*.cu)
TESTCCEXEC=$(subst test/,bin/cc/,$(patsubst %.cc,%.out, $(TESTCCSRC)))
TESTNVEXEC=$(subst test/,bin/cu/,$(patsubst %.cu,%.out, $(TESTNVSRC)))

INCLUDE=src/include
INC=-I./lib/include
LIBS=-L./lib $(INC)
LIB=libClothoids.a #LIB_DUBINS
MORE_FLAGS=

src/obj/cc/%.o: src/%.cc
	$(CC) $(CCFLAGS) $(MORE_FLAGS) -c -o $@ $< $(LIBS)

src/obj/cu/%.o: src/%.cu
	$(NV) $(NVFLAGS) $(MORE_FLAGS) -c -o $@ $< $(LIBS)

bin/cc/%.out: test/%.cc
	$(CC) $(CCFLAGS) $(MORE_FLAGS) -o $@ $< $(LIBS)

bin/cu/%.out: test/%.cu
	$(NV) $(NVFLAGS) $(MORE_FLAGS) -o $@ $< $(LIBS)

all: echo lib $(TESTCCEXEC)

echo:
	@echo "CCSRC: " $(CCSRC)
	@echo "CUSRC: " $(CUSRC)
	@echo "CCOBJ: " $(CCOBJ)
	@echo "CUOBJ: " $(CUOBJ)
	@echo "TESTCCSRC: " $(TESTCCSRC)
	@echo "TESTCUSRC: " $(TESTCUSRC)
	@echo "TESTCCEXEC: " $(TESTCCEXEC)
	@echo "TESTCUEXEC: " $(TESTCUEXEC)

lib: lib/$(LIB)

mvlib:
	@rm -rf lib/include
	$(MKDIR) lib
	$(MKDIR) lib/include
	cp -f $(INCLUDE)/*.hh lib/include

lib/$(LIB): mvlib obj/ bin/ $(CCOBJ) #TODO add CUDA support
	$(AR) lib/$(LIB) $(CCOBJ)

clean_lib:
	rm -rf lib/

clean_obj:
	rm -rf src/obj/

clean_bin:
	rm -rf bin

clean: clean_lib clean_bin clean_obj

run:
	bin/cc/main.out

bin/:
	$(MKDIR) bin/cc
	$(MKDIR) bin/cu
obj/:
	$(MKDIR) src/obj/cc
	$(MKDIR) src/obj/cu

