#include <vector>
#include <queue>
#include "basic.h"

using namespace std;

double findAugmentingPath(Graph &g, vector<double> &curNodeCap, vector<int> &parents, vector<vector<double>> &flowMatrix,
                          int s, int t) {
    fill(parents.begin(), parents.end(), -1);
    fill(curNodeCap.begin(), curNodeCap.end(), 0);
    parents[s] = s;
    curNodeCap[s] = 1e20;
    queue<int> q;
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v = 0; v < g.nodeNum; ++v) {
            if (u == v) continue;
            double residual = g.cap[u][v] - flowMatrix[u][v];
            if (residual > 1e-6 && parents[v] == -1) {
                parents[v] = u;
                curNodeCap[v] = min(curNodeCap[u], residual);
                if (v != t) q.push(v);
                else return curNodeCap[t];
            }
        }
    }
    return 0;
}

MaxFlowResult maxFlowSeq(Graph &g, int s, int t) {
    double flow = 0;
    vector<vector<double>> flowMatrix = vector<vector<double>>(g.nodeNum, vector<double>(g.nodeNum, 0));
    vector<int> parents = vector<int>(g.nodeNum);
    vector<double> curNodeCap = vector<double>(g.nodeNum);
    while (true) {
        double curFlow = findAugmentingPath(g, curNodeCap, parents, flowMatrix, s, t);
        if (curFlow < 1e-6) break;
        flow += curFlow;
        int v = t;
        while (v != s) {
            int u = parents[v];
            flowMatrix[u][v] += curFlow;
            flowMatrix[v][u] -= curFlow;
            v = u;
        }
    }
    MaxFlowResult result;
    result.maxFlow = flow;
    result.finalFlows = flowMatrix;
    return result;
}
