IDIR=./
CXX=g++
CXXFLAGS=-I$(IDIR)

.PHONY: clean build

clean:
	rm -f auth.exe

build: *.cpp
	$(CXX) -o auth.exe $(CXXFLAGS) *.cpp

run:
	./auth.exe $(ARGS)