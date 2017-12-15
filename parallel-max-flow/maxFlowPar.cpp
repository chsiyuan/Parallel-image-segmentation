#include <algorithm>
#include "omp.h"
#include "basic.h"
#include "maxFlowPar.h"

bool exploitNewFrontier(Graph &g, vector<int> &parents, vector<double> &curNodeCap, nodesSet &frontier,
                        vector<vector<double>> &curFlow, vector<nodesSet> &frontiersList, int t) {
    int maxThreads = omp_get_max_threads();
    bool foundAugmentingPath = false;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < frontier.cnt; ++i) {
        if (foundAugmentingPath) {
            i = frontier.cnt;
            continue;
        }
        int threadNum = omp_get_thread_num();

        int u = frontier.nodes[i];

        for (int v = 0; v < g.nodeNum && !foundAugmentingPath; ++v) {
            if (u == v) continue;
            double residual = g.cap[u][v] - curFlow[u][v];
            if (residual > 1e-6 && parents[v] == -1 && __sync_bool_compare_and_swap(&parents[v], -1, u)) {
                curNodeCap[v] = min(curNodeCap[u], residual);
                if (v != t) {
                    int index = frontiersList[threadNum].cnt++;
                    frontiersList[threadNum].nodes[index] = v;
                } else {
                    foundAugmentingPath = true;
                }
            }
        }
    }
    frontier.clear();

    if (foundAugmentingPath) {
        return true;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < maxThreads; ++i) {
        int count = frontiersList[i].cnt;
        if (count > 0) {
            int index = __sync_fetch_and_add(&(frontier.cnt), count);
            for (int j = index; j < count + index; j++) {
                frontier.nodes[j] = frontiersList[i].nodes[j - index];
            }
        }
        frontiersList[i].clear();
    }
    return false;
}

double maxFlowBFS(Graph &g, vector<nodesSet> &frontiersList, nodesSet &frontier, vector<double> &curNodeCap,
                  vector<int> &parents, vector<vector<double>> &flowMatrix, int s, int t) {
    int maxThreads = omp_get_max_threads();
    frontier.clear();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < maxThreads; ++i) {
        frontiersList[i].clear();
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < g.nodeNum; ++i) {
        parents[i] = -1;
        curNodeCap[i] = 0;
    }

    parents[s] = s;
    frontier.nodes[frontier.cnt++] = s;
    curNodeCap[s] = 1e20;

    while (frontier.cnt != 0) {
        bool foundPath = exploitNewFrontier(g, parents, curNodeCap, frontier, flowMatrix, frontiersList, t);
        if (foundPath) return curNodeCap[t];
    }
    return 0;
}

MaxFlowResult maxFlowPar(Graph &g, int s, int t) {
    double flow = 0;
    vector<vector<double>> flowMatrix(g.nodeNum, vector<double>(g.nodeNum));
    vector<int> parents(g.nodeNum);
    vector<double> curNodeCap(g.nodeNum);
    unsigned int maxThreads = omp_get_max_threads();

    nodesSet frontier(g.nodeNum);
    vector<nodesSet> frontiersList(maxThreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < maxThreads; ++i) {
        frontiersList[i] = nodesSet(g.nodeNum);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < g.nodeNum; ++i)
        for (int j = 0; j < g.nodeNum; ++j)
            flowMatrix[i][j] = 0;

    while (true) {
        double tempCapacity = maxFlowBFS(g, frontiersList, frontier, curNodeCap, parents, flowMatrix, s, t);
        if (tempCapacity < 1e-6) {
            break;
        }
        flow += tempCapacity;
        int v = t;
        while (v != s) {
            int u = parents[v];
            flowMatrix[u][v] += tempCapacity;
            flowMatrix[v][u] -= tempCapacity;
            v = u;
        }
    }
    MaxFlowResult result;
    result.maxFlow = flow;
    result.finalFlows = flowMatrix;
    return result;
}

