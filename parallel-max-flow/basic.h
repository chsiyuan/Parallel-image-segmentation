#ifndef FINAL_PROJECT_BASIC_H
#define FINAL_PROJECT_BASIC_H

#include <vector>
#include <random>

using namespace std;

class Graph {
public:
    unsigned int nodeNum;
    vector<vector<double>> cap;

    Graph() {
        nodeNum = 0;
    }

    Graph(unsigned int n) {
        nodeNum = n;
        cap = vector<vector<double>>(n, vector<double>(n));
    }

    static Graph randomGraph(unsigned int n, unsigned int edges, double maxCapacity);
};

class MaxFlowResult {
public:
    double maxFlow;
    vector<vector<double>> finalFlows;
};

#endif //FINAL_PROJECT_BASIC_H
