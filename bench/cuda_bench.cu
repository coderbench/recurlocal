#include "recurlocal/cuda_api.h"
#include "recurlocal/planner.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t e,const char* w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}

__global__ void update_state(float* s,std::size_t n,float a,float b,int iters){
    const std::size_t tid=blockIdx.x*blockDim.x+threadIdx.x, stride=blockDim.x*gridDim.x;
    for(std::size_t i=tid;i<n;i+=stride){float x=s[i]; for(int k=0;k<iters;++k)x=fmaf(x,a,b); s[i]=x;}
}
__global__ void checksum_kernel(const float* d,std::size_t n,double* out){
    __shared__ double sm[256]; unsigned t=threadIdx.x; std::size_t i=blockIdx.x*blockDim.x+t;
    sm[t]=(i<n)?(double)d[i]:0.0; __syncthreads();
    for(unsigned s=128;s;s>>=1){if(t<s)sm[t]+=sm[t+s]; __syncthreads();}
    if(t==0) atomicAdd(out,sm[0]);
}

int main(int argc,char** argv){
 try{
    auto mode=recurlocal::parse_mode(argc>1?argv[1]:"baseline");
    int layers=48,tokens=32,iters=2,device=0; std::size_t state_bytes=3ull*1024*1024;
    for(int i=2;i+1<argc;i+=2){std::string k=argv[i],v=argv[i+1];
      if(k=="--layers")layers=std::stoi(v); else if(k=="--tokens")tokens=std::stoi(v);
      else if(k=="--inner-iters")iters=std::stoi(v); else if(k=="--state-bytes")state_bytes=std::stoull(v);
      else if(k=="--device")device=std::stoi(v); else throw std::invalid_argument("unknown option "+k);}
    check(cudaSetDevice(device),"set device"); cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,device),"props");
    const std::size_t per=state_bytes/sizeof(float), total=(std::size_t)layers*per, bytes=total*sizeof(float);
    float* state=nullptr; check(cudaMalloc(&state,bytes),"malloc");
    std::vector<float> host(total); for(std::size_t i=0;i<total;++i)host[i]=(float)((i%1021)*0.0001);
    check(cudaMemcpy(state,host.data(),bytes,cudaMemcpyHostToDevice),"init");
    cudaStream_t compute{},prefetch{}; check(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking),"compute stream"); check(cudaStreamCreateWithFlags(&prefetch,cudaStreamNonBlocking),"prefetch stream");
    recurlocal::PlannerConfig cfg; cfg.mode=mode; recurlocal::CudaLocalityController ctl(device,cfg); check(ctl.bind_streams(compute,prefetch),"bind");
    cudaEvent_t st{},sp{}; check(cudaEventCreate(&st),"event"); check(cudaEventCreate(&sp),"event"); check(cudaEventRecord(st,compute),"start");
    for(int tok=0;tok<tokens;++tok){for(int l=0;l<layers;++l){
      float* cur=state+(std::size_t)l*per; bool hn=l+1<layers; const float* nxt=hn?state+(std::size_t)(l+1)*per:nullptr;
      check(ctl.before_layer(cur,state_bytes,nxt,hn?per:0,hn),"before");
      update_state<<<1024,256,0,compute>>>(cur,per,0.999999f,0.000001f*(l+1),iters); check(cudaGetLastError(),"update");
      check(ctl.after_layer(),"after");}}
    check(cudaEventRecord(sp,compute),"stop"); check(cudaEventSynchronize(sp),"sync"); float ms=0; check(cudaEventElapsedTime(&ms,st,sp),"elapsed");
    double* dsum=nullptr; check(cudaMalloc(&dsum,sizeof(double)),"sum malloc"); check(cudaMemset(dsum,0,sizeof(double)),"sum zero");
    int cb=(int)std::min<std::size_t>((total+255)/256,65535); checksum_kernel<<<cb,256,0,compute>>>(state,total,dsum); check(cudaGetLastError(),"checksum");
    double sum=0; check(cudaMemcpyAsync(&sum,dsum,sizeof(double),cudaMemcpyDeviceToHost,compute),"sum copy"); check(cudaStreamSynchronize(compute),"done");
    std::cout<<std::fixed<<std::setprecision(6)<<"{"
      <<"\"mode\":\""<<recurlocal::to_string(mode)<<"\","<<"\"gpu\":\""<<prop.name<<"\","<<"\"layers\":"<<layers<<","<<"\"state_bytes_per_layer\":"<<state_bytes<<","<<"\"total_state_bytes\":"<<bytes<<","<<"\"tokens\":"<<tokens<<","<<"\"elapsed_ms\":"<<ms<<","<<"\"ms_per_token\":"<<(ms/tokens)<<","<<"\"l2_bytes\":"<<prop.l2CacheSize<<","<<"\"persisting_l2_max_bytes\":"<<prop.persistingL2CacheMaxSize<<","<<"\"actual_l2_set_aside_bytes\":"<<ctl.l2_set_aside_bytes()<<","<<"\"checksum\":"<<sum<<"}\n";
    cudaFree(dsum); ctl.reset(); cudaEventDestroy(st); cudaEventDestroy(sp); cudaStreamDestroy(prefetch); cudaStreamDestroy(compute); cudaFree(state); return 0;
 }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
}
