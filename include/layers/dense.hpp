#pragma once
#include <Eigen/Dense>
#include <random>
#include <cmath>

struct Dense {
Eigen::MatrixXf W; Eigen::VectorXf b; float lr; Eigen::VectorXf x;
Dense(int in_dim, int out_dim, float lr_=0.01f): W(out_dim, in_dim), b(Eigen::VectorXf::Zero(out_dim)), lr(lr_){
std::mt19937 rng(123); float scale=std::sqrt(2.f/in_dim); std::normal_distribution<float> N(0.f, scale);
for(int r=0;r<W.rows();++r) for(int c=0;c<W.cols();++c) W(r,c)=N(rng);
}
Eigen::VectorXf forward(const Eigen::VectorXf &in){ x=in; return W*in + b; }
Eigen::VectorXf backward(const Eigen::VectorXf &g){ Eigen::MatrixXf dW = g * x.transpose(); Eigen::VectorXf db=g; Eigen::VectorXf dx=W.transpose()*g; W -= lr*dW; b -= lr*db; return dx; }
};