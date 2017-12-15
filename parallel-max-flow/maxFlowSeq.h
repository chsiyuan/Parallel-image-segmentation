#ifndef FINAL_PROJECT_MAXFLOWSEQ_H
#define FINAL_PROJECT_MAXFLOWSEQ_H
#include "basic.h"

double findAugmentingPath(Graph &g, vector<double> &curNodeCap, vector<int> &parents, vector<vector<double>> &flowMatrix,
                          int s, int t);

MaxFlowResult maxFlowSeq(Graph &g, int s, int t);
#endif //FINAL_PROJECT_MAXFLOWSEQ_H
