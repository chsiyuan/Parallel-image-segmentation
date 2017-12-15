main: main.o maxFlowPar.o maxFlowSeq.o basic.o
	g++ -o main main.o maxFlowPar.o maxFlowSeq.o basic.o -fopenmp -std=c++11

main.o: main.cpp basic.h maxFlowPar.h maxFlowSeq.h
	g++ -c main.cpp -fopenmp -std=c++11

maxFlowPar.o: maxFlowPar.cpp maxFlowPar.h basic.h
	g++ -c maxFlowPar.cpp -fopenmp -std=c++11

maxFlowSeq.o: maxFlowSeq.cpp maxFlowSeq.h basic.h
	g++ -c maxFlowSeq.cpp -fopenmp -std=c++11

basic.o: basic.cpp basic.h
	g++ -c basic.cpp -fopenmp -std=c++11

clean:
	rm main main.o maxFlowPar.o maxFlowSeq.o
.PHONY: clean
