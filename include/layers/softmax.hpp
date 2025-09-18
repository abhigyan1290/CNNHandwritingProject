

#pragma once
#include <Eigen/Dense>
#include <cmath>


struct SoftmaxCE {
Eigen::VectorXf probs; int target=-1;
float forward(const Eigen::VectorXf &logits, int y){ target=y; float m=logits.maxCoeff(); Eigen::VectorXf z=(logits.array()-m).exp(); probs=z/z.sum(); return -std::log(std::max(1e-9f, probs(y))); }
Eigen::VectorXf backward(){ Eigen::VectorXf g=probs; g(target) -= 1.f; return g; }
};