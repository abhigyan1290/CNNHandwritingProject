#pragma once
#include <Eigen/Dense>
#include <vector>


struct MaxPool2x2 {
    std::vector<Eigen::MatrixXf> mask; int inH=0,inW=0;
    std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf> &in){
    inH=in[0].rows(); inW=in[0].cols(); mask.assign(in.size(), Eigen::MatrixXf::Zero(inH,inW));
    int H=inH/2, W=inW/2; std::vector<Eigen::MatrixXf> out(in.size(), Eigen::MatrixXf::Zero(H,W));
    for(size_t ch=0; ch<in.size(); ++ch){
        for(int r=0;r<H;++r) for(int c=0;c<W;++c){
        auto block=in[ch].block(r*2,c*2,2,2); float m; Eigen::Index rr,cc; m=block.maxCoeff(&rr,&cc);
        out[ch](r,c)=m; mask[ch](r*2+rr, c*2+cc)=1.f;
        }
    }
    return out;
}

std::vector<Eigen::MatrixXf> backward(const std::vector<Eigen::MatrixXf> &g){
        int H=inH/2, W=inW/2; std::vector<Eigen::MatrixXf> dx(mask.size(), Eigen::MatrixXf::Zero(inH,inW));
        for(size_t ch=0; ch<mask.size(); ++ch)
            for(int r=0;r<H;++r) for(int c=0;c<W;++c)
                for(int br=0;br<2;++br) for(int bc=0;bc<2;++bc){ int rr=r*2+br, cc=c*2+bc; if(mask[ch](rr,cc)>0.5f) dx[ch](rr,cc)=g[ch](r,c); }
        return dx;
    }
};