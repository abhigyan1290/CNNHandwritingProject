
#pragma once
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>


struct Sampler {
std::vector<int> idx; size_t p=0; std::mt19937 rng{1337};
explicit Sampler(int n){ idx.resize(n); std::iota(idx.begin(), idx.end(), 0); std::shuffle(idx.begin(), idx.end(), rng);}
int next(){ if(p>=idx.size()){ std::shuffle(idx.begin(), idx.end(), rng); p=0;} return idx[p++]; }
};