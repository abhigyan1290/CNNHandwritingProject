#pragma once
#include <Eigen/Dense>
#include <vector>
#include "layers/conv2d.hpp"
#include "layers/relu.hpp"
#include "layers/maxpool2x2.hpp"
#include "layers/flatten.hpp"
#include "layers/dense.hpp"
#include "layers/softmax.hpp"


struct CNN {
Conv2D c1; ReLU2D r1; MaxPool2x2 p1;
Conv2D c2; ReLU2D r2; MaxPool2x2 p2;
Flatten fl; Dense fc1; ReLU1D r3; Dense fc2; SoftmaxCE ce;


CNN(float lr): c1(1,8,3,lr), c2(8,16,3,lr), fc1(16*7*7,128,lr), fc2(128,10,lr) {}


Eigen::VectorXf forward(const Eigen::MatrixXf &img){
std::vector<Eigen::MatrixXf> x0={img};
auto x1=c1.forward(x0); x1=r1.forward(x1); x1=p1.forward(x1);
auto x2=c2.forward(x1); x2=r2.forward(x2); x2=p2.forward(x2);
Eigen::VectorXf v=fl.forward(x2);
Eigen::VectorXf h=fc1.forward(v);
Eigen::VectorXf h_relu=r3.forward(h);
return fc2.forward(h_relu);
}


float train_step(const Eigen::MatrixXf &img, int y){
Eigen::VectorXf logits=forward(img);
float loss=ce.forward(logits,y);
auto g=ce.backward();
auto g2=fc2.backward(g);
auto g3=r3.backward(g2);
auto g4=fc1.backward(g3);
auto g5=fl.backward(g4);
auto g6=p2.backward(g5);
auto g7=r2.backward(g6);
auto g8=c2.backward(g7);
auto g9=p1.backward(g8);
auto g10=r1.backward(g9); (void)g10; // end of chain
return loss;
}


int predict(const Eigen::MatrixXf &img){ auto l=forward(img); Eigen::Index idx; l.maxCoeff(&idx); return (int)idx; }
};