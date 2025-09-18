#pragma once
#include <Eigen/Dense>
#include <vector>


struct ReLU2D {
std::vector<Eigen::MatrixXf> x;
std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf> &in){ x=in; auto y=in; for(auto &m: y) m = m.cwiseMax(0.f); return y; }
std::vector<Eigen::MatrixXf> backward(const std::vector<Eigen::MatrixXf> &g){ auto dx=g; for(size_t i=0;i<dx.size();++i){ auto mask=(x[i].array()>0).cast<float>(); dx[i]=dx[i].cwiseProduct(mask.matrix()); } return dx; }
};


struct ReLU1D { // for vectors
Eigen::VectorXf x;
Eigen::VectorXf forward(const Eigen::VectorXf &in){ x=in; return in.cwiseMax(0.f); }
Eigen::VectorXf backward(const Eigen::VectorXf &g){ Eigen::VectorXf mask=(x.array()>0).cast<float>(); return g.cwiseProduct(mask); }
};