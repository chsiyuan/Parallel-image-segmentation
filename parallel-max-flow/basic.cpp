#include "basic.h"
#include <algorithm>

Graph Graph:: randomGraph(unsigned int n, unsigned int edges, double maxCapacity) {
    Graph g(n);
    vector<int> path(n - 1);
    for (int i = 1; i < n; ++i) path[i - 1] = i;
    random_shuffle(path.begin(), path.end());
    int t = n - 1;
    int fst = path[0];
    double upper_bound = maxCapacity;
    uniform_real_distribution<double> unif(0, upper_bound);
    default_random_engine re;

    g.cap[0][fst] = unif(re);
    int remain = edges - 1;
    int last = fst;
    for (auto it = (++path.begin()); it != path.end() && *it != t; ++it) {
        g.cap[last][*it] = unif(re);
        last = *it;
        --remain;
    }
    g.cap[last][t] = unif(re);
    --remain;
    for (int i = 0; i < remain;) {
        int j = rand() % g.nodeNum;
        int k = rand() % g.nodeNum;
        if (j != k) {
            g.cap[j][k] = unif(re);
            ++i;
        }
    }
    return g;
}
