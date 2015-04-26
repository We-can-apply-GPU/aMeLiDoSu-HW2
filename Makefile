all: pre.out genTest.out
pre.out: userCode/pre.cpp
	g++ userCode/pre.cpp -std=c++11 -o pre.out
genTest.out: userCode/genTest.cpp
	g++ userCode/genTest.cpp -std=c++11 -o genTest.out
