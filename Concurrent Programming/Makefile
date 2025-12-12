IDIR=./
CXX=g++
CXXFLAGS=-I$(IDIR)

.PHONY: clean build

clean:
	rm -f assignment.exe

build: *.cpp
	$(CXX) -o assignment.exe $(CXXFLAGS) *.cpp

run:
	./assignment.exe $(ARGS)