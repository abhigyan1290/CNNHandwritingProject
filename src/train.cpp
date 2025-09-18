

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "utils/mnist_io.hpp"
#include "utils/sampler.hpp"
#include "model.hpp"


#ifdef OPENCV_OK
#include <opencv2/opencv.hpp>
#endif


using namespace std;


static float eval_acc(CNN &net, const utils::MNIST &ds, int limit=-1){
int N=(limit>0? std::min(limit,(int)ds.images.size()): (int)ds.images.size());
int ok=0; for(int i=0;i<N;++i) if(net.predict(ds.images[i])==ds.labels[i]) ++ok; return 100.f*ok/N;
}


int main(int argc, char** argv){
if(argc<5){ cerr << "Usage: ./mnist_cnn train-images train-labels test-images test-labels\n"; return 1; }
auto train=utils::load_idx(argv[1], argv[2]);
auto test =utils::load_idx(argv[3], argv[4]);


const int EPOCHS=2; const float LR=0.01f; const int STEPS=60000; // tweak as desired
CNN net(LR);


#ifdef OPENCV_OK
// quick visual sanity
cv::Mat show(28,28,CV_32F, const_cast<float*>(train.images[0].data()));
cv::Mat disp; show.convertTo(disp, CV_8U, 255.0); cv::resize(disp, disp, {140,140},0,0, cv::INTER_NEAREST); cv::imshow("sample", disp); cv::waitKey(5);
#endif


Sampler samp((int)train.images.size());
for(int ep=1; ep<=EPOCHS; ++ep){
double avg=0; int cnt=0;
for(int s=0; s<STEPS; ++s){
int i=samp.next();
float loss=net.train_step(train.images[i], train.labels[i]);
avg+=loss; ++cnt;
if((s+1)%1000==0){ cout << "[ep "<<ep<<"] step "<<(s+1)<<" avg_loss="<<(avg/cnt) <<"\n"; avg=0; cnt=0; }
}
cout << "Test acc (1k): " << eval_acc(net, test, 1000) << "%\n";
}
cout << "Final test acc: " << eval_acc(net, test) << "%\n";
return 0;
}