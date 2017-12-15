#include <vector>
#include <iostream>
#include "basic.h"
#include "maxFlowSeq.h"
#include "omp.h"
#include "maxFlowPar.h"
using namespace std;

void printTime(double begin, string prefix, unsigned int nodeNum) {
    cout << "*********" << prefix << " node num: " << nodeNum << "********" << endl;
    cout << "time: " << omp_get_wtime() - begin << " s" << endl;
}

int main(int argc, char *argv[]) {

    string mode;
    if (argc == 1)
        mode = "seq";
    else {
        mode = "par";
    }
    vector<unsigned int> nodeNumList = {400, 800, 1600, 3200, 6400, 12000};
    if (mode == "seq") {
        for (auto num : nodeNumList) {
            unsigned int nodeNum = num, edges = num * 20;
            double maxCapacity = 20;
            Graph g = Graph::randomGraph(nodeNum, edges, maxCapacity);
            double begin = omp_get_wtime();
            MaxFlowResult res = maxFlowSeq(g, 0, nodeNum - 1);
            printTime(begin, "sequential", nodeNum);
        }
    } else {
        for (auto num : nodeNumList) {
            unsigned int nodeNum = num, edges = nodeNum * 20;
            double maxCapacity = 20;
            Graph g = Graph::randomGraph(nodeNum, edges, maxCapacity);
            double begin = omp_get_wtime();
            MaxFlowResult res = maxFlowPar(g, 0, nodeNum - 1);
            printTime(begin, "parallel", nodeNum);
        }
    }
    return 0;
}

