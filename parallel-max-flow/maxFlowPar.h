#ifndef FINAL_PROJECT_MAXFLOWPAR_H
#define FINAL_PROJECT_MAXFLOWPAR_H

#include <vector>

using namespace std;

class nodesSet {
public:
    int cnt;
    int maxNodes;
    vector<int> nodes;

    nodesSet() {
        cnt = 0;
        maxNodes = 0;
    }

    nodesSet(int cnt) {
        maxNodes = cnt;
        nodes = vector<int>(cnt);
        clear();
    }

    void clear() {
        this->cnt = 0;
    }

};

bool exploitNewFrontier(Graph &g, vector<int> &parents, vector<double> &curNodeCap, nodesSet &frontier,
                        vector<vector<double>> &curFlow, vector<nodesSet> &frontiersList, int t);

double maxFlowBFS(Graph &g, vector<nodesSet> &frontiersList, nodesSet &frontier, vector<double> &curNodeCap,
                  vector<int> &parents, vector<vector<double>> &flowMatrix, int s, int t);

MaxFlowResult maxFlowPar(Graph &g, int s, int t);
#endif //FINAL_PROJECT_MAXFLOWPAR_H
