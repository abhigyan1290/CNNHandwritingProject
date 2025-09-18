#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>


struct Conv2D {
int in_ch, out_ch, k; float lr;
std::vector<std::vector<Eigen::MatrixXf>> W; // [out][in] kxk
Eigen::VectorXf b; // [out]


std::vector<Eigen::MatrixXf> x_in;

Conv2D(int in_, int out_, int k_, float lr_=0.01f)
    : in_ch(in_), out_ch(out_), k(k_), lr(lr_), b(Eigen::VectorXf::Zero(out_)){
    std::mt19937 rng(42); float scale = std::sqrt(2.f/(in_*k_*k_));
    std::normal_distribution<float> N(0.f, scale);
    W.resize(out_, std::vector<Eigen::MatrixXf>(in_, Eigen::MatrixXf::Zero(k_,k_)));
    for(int o=0;o<out_;++o) for(int i=0;i<in_;++i)
        for(int r=0;r<k_;++r) for(int c=0;c<k_;++c) W[o][i](r,c)=N(rng);
}


static Eigen::MatrixXf pad_same(const Eigen::MatrixXf &m, int pad){
Eigen::MatrixXf o = Eigen::MatrixXf::Zero(m.rows()+2*pad, m.cols()+2*pad);
o.block(pad,pad,m.rows(),m.cols()) = m; return o;
}


std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf> &x){
    x_in = x; int H=x[0].rows(), Wd=x[0].cols(), pad=k/2;
    std::vector<Eigen::MatrixXf> xp(in_ch);
    for(int i=0;i<in_ch;++i) xp[i]=pad_same(x[i], pad);
    std::vector<Eigen::MatrixXf> y(out_ch, Eigen::MatrixXf::Zero(H,Wd));
    for(int o=0;o<out_ch;++o){
        Eigen::MatrixXf acc = Eigen::MatrixXf::Constant(H,Wd,b(o));
        for(int i=0;i<in_ch;++i){
            for(int r=0;r<H;++r) for(int c=0;c<Wd;++c)
                acc(r,c) += (xp[i].block(r,c,k,k).cwiseProduct(W[o][i])).sum();
        }
        y[o]=acc;
}
return y;
}

std::vector<Eigen::MatrixXf> backward(const std::vector<Eigen::MatrixXf> &gout){
    int H=x_in[0].rows(), Wd=x_in[0].cols(), pad=k/2;
    std::vector<std::vector<Eigen::MatrixXf>> dW(out_ch, std::vector<Eigen::MatrixXf>(in_ch, Eigen::MatrixXf::Zero(k,k)));
    Eigen::VectorXf db = Eigen::VectorXf::Zero(out_ch);
    std::vector<Eigen::MatrixXf> dx(in_ch, Eigen::MatrixXf::Zero(H,Wd));
    std::vector<Eigen::MatrixXf> xp(in_ch);
    for(int i=0;i<in_ch;++i) xp[i]=pad_same(x_in[i], pad);


    for(int o=0;o<out_ch;++o){
        db(o) += gout[o].sum();
        for(int i=0;i<in_ch;++i)
            for(int r=0;r< H;++r) for(int c=0;c<Wd;++c)
                dW[o][i] += gout[o](r,c)*xp[i].block(r,c,k,k);
    }
    std::vector<Eigen::MatrixXf> go_p(out_ch);
    for(int o=0;o<out_ch;++o) go_p[o]=pad_same(gout[o], pad);
    for(int i=0;i<in_ch;++i){
        Eigen::MatrixXf acc=Eigen::MatrixXf::Zero(H,Wd);
        for(int o=0;o<out_ch;++o){
            Eigen::MatrixXf Kf = W[o][i].colwise().reverse().rowwise().reverse();
            for(int r=0;r<H;++r) for(int c=0;c<Wd;++c)
                acc(r,c) += (go_p[o].block(r,c,k,k).cwiseProduct(Kf)).sum();
    }
    dx[i]=acc;
    }
for(int o=0;o<out_ch;++o){ b(o) -= lr*db(o); for(int i=0;i<in_ch;++i) W[o][i] -= lr*dW[o][i]; }
    return dx;
}

};