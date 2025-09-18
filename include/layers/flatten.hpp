#pragma once
#include <Eigen/Dense>
#include <vector>

struct Flatten {
int C=0,H=0,W=0;
Eigen::VectorXf forward(const std::vector<Eigen::MatrixXf> &in){ C=in.size(); H=in[0].rows(); W=in[0].cols(); Eigen::VectorXf v(C*H*W); int off=0; for(int c=0;c<C;++c){ Eigen::Map<const Eigen::VectorXf> vv(in[c].data(), H*W); v.segment(off,H*W)=vv; off+=H*W;} return v; }
std::vector<Eigen::MatrixXf> backward(const Eigen::VectorXf &g){ std::vector<Eigen::MatrixXf> out(C, Eigen::MatrixXf::Zero(H,W)); int off=0; for(int c=0;c<C;++c){ Eigen::Map<const Eigen::MatrixXf> mm(g.data()+off, H, W); out[c]=mm; off+=H*W;} return out; }
};