#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>


namespace utils {


inline uint32_t read_be_u32(std::ifstream &ifs){
uint32_t x; ifs.read(reinterpret_cast<char*>(&x),4);
#if defined(_MSC_VER)
    return _byteswap_ulong(x);
#else
    return __builtin_bswap32(x);
#endif
}


struct MNIST {
std::vector<Eigen::MatrixXf> images; // 28x28 in [0,1]
std::vector<uint8_t> labels; // 0..9
};


inline MNIST load_idx(const std::string &img_path, const std::string &lbl_path){
std::ifstream fi(img_path, std::ios::binary); if(!fi) throw std::runtime_error("open images failed");
std::ifstream fl(lbl_path, std::ios::binary); if(!fl) throw std::runtime_error("open labels failed");


uint32_t magic_i = read_be_u32(fi);
uint32_t n = read_be_u32(fi);
uint32_t rows = read_be_u32(fi);
uint32_t cols = read_be_u32(fi);
if(magic_i != 2051 || rows!=28 || cols!=28) throw std::runtime_error("bad image idx");


uint32_t magic_l = read_be_u32(fl);
uint32_t nl = read_be_u32(fl);
if(magic_l != 2049 || nl != n) throw std::runtime_error("bad label idx");


MNIST ds; ds.images.reserve(n); ds.labels.resize(n);
std::vector<uint8_t> buf(28*28);
for(uint32_t i=0;i<n;++i){
fi.read(reinterpret_cast<char*>(buf.data()), buf.size());
Eigen::MatrixXf img(28,28);
for(int r=0;r<28;++r) for(int c=0;c<28;++c) img(r,c) = buf[r*28+c]/255.0f;
ds.images.emplace_back(std::move(img));
uint8_t lab; fl.read(reinterpret_cast<char*>(&lab),1); ds.labels[i]=lab;
}
return ds;
}


} // namespace utils