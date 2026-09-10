#include "tensortransit/cuda_recurrent.h"
#include "tensortransit/recurrent.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t e,const char* w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}

// State layout. The update touches every element exactly once whichever layout is chosen,
// so the map must be a bijection and the final state is identical; only the ACCESS ORDER
// changes, which is the whole experiment. Real recurrent state is [head][dv][dk] and a
// kernel that walks it in a different order than it is stored pays for it in coalescing
// and in how much of each 128-byte line is used.
enum Layout { kLinear=0, kHeadInterleaved=1, kTileSwapped=2 };

__device__ __forceinline__ std::size_t map_index(std::size_t i,int layout,std::size_t n){
    if(layout==kHeadInterleaved){
        // Consecutive threads land in different heads: the same bytes, fully uncoalesced.
        constexpr std::size_t heads=48;
        const std::size_t per_head=n/heads;
        if(per_head==0) return i;
        const std::size_t body=per_head*heads;
        if(i>=body) return i;                      // remainder keeps identity
        return (i%heads)*per_head + (i/heads);
    }
    if(layout==kTileSwapped){
        // Adjacent 128-byte tiles swapped: lines stay intact, sequential prefetchability
        // does not. Always a bijection.
        constexpr std::size_t tile=32;
        const std::size_t tiles=n/tile;
        if(tiles<2) return i;
        const std::size_t body=tiles*tile;
        if(i>=body) return i;
        const std::size_t t=i/tile,o=i%tile;
        const std::size_t st=(t^1u)<tiles?(t^1u):t;
        return st*tile+o;
    }
    return i;
}

// Each element depends only on itself, so the final state is independent of scheduling and
// must be bit-identical across every mode, layout, strategy and distance.
// When `next` is non-null this kernel also performs the pre-touch itself instead of a
// separate stream doing it: kernel-integrated prefetch, with no extra launch and no
// second stream to schedule.
__global__ void update_state(float* s,std::size_t n,float a,float b,int iters,int layout,
                             const float* __restrict__ next,std::size_t next_count,float* sink){
    const std::size_t tid=blockIdx.x*blockDim.x+threadIdx.x, stride=(std::size_t)blockDim.x*gridDim.x;
    for(std::size_t i=tid;i<n;i+=stride){
        const std::size_t p=map_index(i,layout,n);
        float x=s[p]; for(int k=0;k<iters;++k)x=fmaf(x,a,b); s[p]=x;
    }
    if(next){
        float acc=0.0f;
        for(std::size_t i=tid;i<next_count;i+=stride) acc+=__ldcg(&next[i]);
        if(threadIdx.x==0&&blockIdx.x==0) *sink=acc;
    }
}

// Checksum in two deterministic stages. Stage one uses a grid-stride loop over a
// fixed grid, so every element is covered whatever the state size and the
// per-thread summation order is fixed; stage two folds the partials in index
// order. A single-stage atomicAdd reduction sized as (n+255)/256 blocks would be
// clamped to the 65535 grid limit and silently skip most of the default geometry.
constexpr int kChecksumBlocks=1024;
constexpr int kChecksumThreads=256;

// Stands in for the traffic between recurrent layers in a real hybrid model: attention and
// MoE weights streaming through the same L2 the recurrent state is trying to stay resident
// in. Without it the persisting window has nothing to defend against, which is exactly the
// interference the project exists to manage.
// Two shapes of interference, because they are not the same problem:
//   reuse    - every layer reads the same buffer. Models a shared table or expert that is
//              re-read constantly, so it is genuinely hot and wants to stay resident.
//   distinct - each layer reads its own slice, so the whole buffer is walked once per token.
//              Models per-layer attention/MoE weights, which are streamed and never reused
//              within a token. This is the traffic a persisting window should be protected
//              from, and marking it streaming should be free.
__global__ void stream_interference(const float* __restrict__ w, std::size_t n, float* __restrict__ sink){
    const std::size_t tid=blockIdx.x*blockDim.x+threadIdx.x, stride=(std::size_t)blockDim.x*gridDim.x;
    float acc=0.0f;
    for(std::size_t i=tid;i<n;i+=stride) acc+=w[i];
    if(threadIdx.x==0 && blockIdx.x==0) *sink=acc;
}

__global__ void checksum_partials(const float* __restrict__ d,std::size_t n,double* __restrict__ partial){
    __shared__ double sm[kChecksumThreads];
    const unsigned t=threadIdx.x;
    const std::size_t stride=(std::size_t)blockDim.x*gridDim.x;
    double acc=0.0;
    for(std::size_t i=(std::size_t)blockIdx.x*blockDim.x+t;i<n;i+=stride) acc+=(double)d[i];
    sm[t]=acc; __syncthreads();
    for(unsigned s=blockDim.x/2;s;s>>=1){if(t<s)sm[t]+=sm[t+s]; __syncthreads();}
    if(t==0) partial[blockIdx.x]=sm[0];
}
__global__ void checksum_fold(const double* __restrict__ partial,int count,double* __restrict__ out){
    double acc=0.0; for(int i=0;i<count;++i) acc+=partial[i]; *out=acc;
}

int main(int argc,char** argv){
 try{
    auto mode=tensortransit::parse_mode(argc>1?argv[1]:"baseline");
    int layers=48,tokens=32,warmup=4,iters=2,device=0,distance=1,sequences=1; std::size_t state_bytes=3ull*1024*1024;
    std::size_t stream_bytes=0;
    auto pre_touch=tensortransit::PreTouchStrategy::Vec4;
    auto hot_set_policy=tensortransit::HotSetPolicy::Proportional;
    auto set_aside_policy=tensortransit::SetAsidePolicy::Fixed;
    // Defaults to the v0.1 accounting, so an existing command line still measures what it did.
    auto hot_set_model=tensortransit::HotSetModel::CurrentLayer;
    auto schedule=tensortransit::PrefetchSchedule::Uniform;
    int layout=kLinear; bool fused=false, qos=false, stream_distinct=false;
    for(int i=2;i<argc;i+=2){
      if(i+1>=argc) throw std::invalid_argument(std::string("option ")+argv[i]+" needs a value");
      std::string k=argv[i],v=argv[i+1];
      if(k=="--layers")layers=std::stoi(v); else if(k=="--tokens")tokens=std::stoi(v);
      else if(k=="--warmup-tokens")warmup=std::stoi(v);
      else if(k=="--inner-iters")iters=std::stoi(v); else if(k=="--state-bytes")state_bytes=std::stoull(v);
      else if(k=="--device")device=std::stoi(v);
      else if(k=="--prefetch-distance")distance=std::stoi(v);
      else if(k=="--pre-touch")pre_touch=tensortransit::parse_pre_touch_strategy(v.c_str());
      else if(k=="--sequences")sequences=std::stoi(v);
      else if(k=="--stream-bytes")stream_bytes=std::stoull(v);
      else if(k=="--hot-set-policy")hot_set_policy=tensortransit::parse_hot_set_policy(v.c_str());
      else if(k=="--set-aside-policy")set_aside_policy=tensortransit::parse_set_aside_policy(v.c_str());
      else if(k=="--hot-set-model")hot_set_model=tensortransit::parse_hot_set_model(v.c_str());
      else if(k=="--prefetch-schedule")schedule=tensortransit::parse_prefetch_schedule(v.c_str());
      else if(k=="--prefetch-impl"){ if(v=="fused")fused=true; else if(v=="stream")fused=false;
                                     else throw std::invalid_argument("--prefetch-impl must be stream or fused"); }
      else if(k=="--state-layout"){ if(v=="linear")layout=kLinear; else if(v=="head_interleaved")layout=kHeadInterleaved;
                                    else if(v=="tile_swapped")layout=kTileSwapped;
                                    else throw std::invalid_argument("unknown --state-layout "+v); }
      else if(k=="--qos"){ if(v=="on")qos=true; else if(v=="off")qos=false;
                           else throw std::invalid_argument("--qos must be on or off"); }
      else if(k=="--stream-mode"){ if(v=="distinct")stream_distinct=true; else if(v=="reuse")stream_distinct=false;
                                   else throw std::invalid_argument("--stream-mode must be reuse or distinct"); }
      else throw std::invalid_argument("unknown option "+k);}
    if(layers<1) throw std::invalid_argument("--layers must be >= 1");
    if(tokens<1) throw std::invalid_argument("--tokens must be >= 1");
    if(warmup<0) throw std::invalid_argument("--warmup-tokens must be >= 0");
    if(iters<0) throw std::invalid_argument("--inner-iters must be >= 0");
    if(!state_bytes||state_bytes%sizeof(float)) throw std::invalid_argument("--state-bytes must be a non-zero multiple of 4");
    if(distance<0||distance>tensortransit::kMaxPrefetchDistance) throw std::invalid_argument("--prefetch-distance must be in [0,8]");
    if(sequences<1) throw std::invalid_argument("--sequences must be >= 1");
    if(stream_bytes%sizeof(float)) throw std::invalid_argument("--stream-bytes must be a multiple of 4");

    check(cudaSetDevice(device),"set device"); cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,device),"props");
    // Provenance: a timing without the driver/toolkit/device it came from cannot be
    // reproduced or compared later (overview section 18).
    int driver_version=0, runtime_version=0; cudaDriverGetVersion(&driver_version); cudaRuntimeGetVersion(&runtime_version);
    // Batched decode lays the per-layer state of every active sequence out together, so the
    // hot set at a layer scales with concurrency. That is what makes the hot-set policy
    // measurable: at 48 MiB of set-aside it starts oversubscribing past 16 sequences.
    const std::size_t per=state_bytes/sizeof(float);
    const std::size_t block=per*(std::size_t)sequences;            // one layer, all sequences
    const std::size_t block_bytes=block*sizeof(float);
    const std::size_t total=(std::size_t)layers*block, bytes=total*sizeof(float);
    float* state=nullptr; check(cudaMalloc(&state,bytes),"malloc");
    float* weights=nullptr; float* sink=nullptr; std::size_t stream_count=stream_bytes/sizeof(float);
    if(stream_count){ check(cudaMalloc(&weights,stream_count*sizeof(float)),"stream malloc");
                      check(cudaMemset(weights,0,stream_count*sizeof(float)),"stream init"); }
    check(cudaMalloc(&sink,sizeof(float)),"sink malloc");   // fused pre-touch also needs it
    std::vector<float> host(total); for(std::size_t i=0;i<total;++i)host[i]=(float)((i%1021)*0.0001);
    check(cudaMemcpy(state,host.data(),bytes,cudaMemcpyHostToDevice),"init");
    cudaStream_t compute{},prefetch{}; check(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking),"compute stream"); check(cudaStreamCreateWithFlags(&prefetch,cudaStreamNonBlocking),"prefetch stream");
    cudaEvent_t st{},sp{},pf{}; check(cudaEventCreate(&st),"event"); check(cudaEventCreate(&sp),"event"); check(cudaEventCreate(&pf),"event");

    float ms=0.0f; double sum=0.0; std::size_t set_aside=0; tensortransit::ControllerStats stats{};
    { // The controller clears its access-policy window on destruction, so it has
      // to outlive nothing: it must die while the compute stream is still valid.
    // Kernel-integrated prefetch is done by the compute kernel, so the controller must not
    // also issue it on the prefetch stream; it keeps only the persisting half of the mode.
    const bool mode_prefetches = mode==tensortransit::LocalityMode::Prefetch || mode==tensortransit::LocalityMode::Combined;
    const bool mode_persists = mode==tensortransit::LocalityMode::Persist || mode==tensortransit::LocalityMode::Combined;
    const bool fused_active = fused && mode_prefetches;
    tensortransit::PlannerConfig cfg;
    cfg.mode = fused_active ? (mode_persists ? tensortransit::LocalityMode::Persist
                                             : tensortransit::LocalityMode::Baseline) : mode;
    cfg.prefetch_distance=distance; cfg.pre_touch=pre_touch;
    cfg.hot_set_policy=hot_set_policy; cfg.prefetch_schedule=schedule;
    cfg.set_aside_policy=set_aside_policy;
    cfg.hot_set_model=hot_set_model;
    tensortransit::CudaLocalityController ctl(device,cfg);
    check(ctl.status(),"controller init"); check(ctl.bind_streams(compute,prefetch),"bind");
    set_aside=ctl.l2_set_aside_bytes();

    tensortransit::RecurrentGeometry geometry;
    geometry.recurrent_layers=layers;
    geometry.bytes_per_layer=state_bytes;          // one layer, ONE sequence
    geometry.sequences=sequences;
    // Distinct mode walks the streaming buffer once per token; reuse mode has every layer
    // re-read all of it, so the bytes passing through L2 between two visits to the same
    // state differ by a factor of `layers`.
    geometry.streamed_bytes_per_token=stream_distinct?stream_bytes:stream_bytes*(std::size_t)layers;

    auto run_token=[&]{ ctl.begin_sequence(); for(int l=0;l<layers;++l){
      // The runtime owns what "next" means; the planner only decides whether and how far.
      float* cur=state+(std::size_t)l*block;
      const int ahead=distance>0?distance:1;
      const bool hn=l+ahead<layers;
      const float* nxt=hn?state+(std::size_t)(l+ahead)*block:nullptr;
      // The geometry is what says every layer's state is revisited within a token. That is
      // exactly what the CurrentLayer model cannot see, and what let `persist` measure
      // negative at four sequences while the planner reported no oversubscription at all.
      tensortransit::StateSegment cur_seg{cur,block_bytes,state,bytes,tensortransit::StateKind::Matrix};
      tensortransit::StateSegment next_seg{nxt,hn?block_bytes:0,state,bytes,tensortransit::StateKind::Matrix};
      check(ctl.before_layer(&cur_seg,1,hn?&next_seg:nullptr,hn?1:0,hn,geometry),"before");
      const float* fnext = (fused_active&&hn)?nxt:nullptr;
      update_state<<<1024,256,0,compute>>>(cur,block,0.999999f,0.000001f*(l+1),iters,layout,
                                           fnext,fnext?block:0,sink); check(cudaGetLastError(),"update");
      check(ctl.after_layer(),"after");
      if(stream_count){
        // In distinct mode each layer walks its own slice, so one token touches the whole
        // buffer exactly once instead of re-reading all of it 48 times.
        const std::size_t slice = stream_distinct ? stream_count/(std::size_t)layers : stream_count;
        const float* w = stream_distinct ? weights+(std::size_t)l*slice : weights;
        if(slice){
          if(qos) check(ctl.before_streaming_region(const_cast<float*>(w),slice*sizeof(float)),"qos");
          stream_interference<<<1024,256,0,compute>>>(w,slice,sink); check(cudaGetLastError(),"stream");
          if(qos) check(ctl.after_streaming_region(),"qos end"); } }}};

    // Untimed warm-up absorbs module load, first-touch page mapping and cold-cache
    // effects that would otherwise dilute a mode-to-mode delta the go/no-go gate
    // has to resolve at a few percent.
    for(int tok=0;tok<warmup;++tok) run_token();
    check(cudaStreamSynchronize(compute),"warmup compute"); check(cudaStreamSynchronize(prefetch),"warmup prefetch");

    check(cudaEventRecord(st,compute),"start");
    for(int tok=0;tok<tokens;++tok) run_token();
    // Join the prefetch stream before stopping the clock. Pre-touch is a cost this
    // benchmark exists to weigh, so it must not run off the measured region.
    check(cudaEventRecord(pf,prefetch),"prefetch mark"); check(cudaStreamWaitEvent(compute,pf,0),"join prefetch");
    check(cudaEventRecord(sp,compute),"stop"); check(cudaEventSynchronize(sp),"sync"); check(cudaEventElapsedTime(&ms,st,sp),"elapsed");

    double *dpart=nullptr,*dsum=nullptr;
    check(cudaMalloc(&dpart,kChecksumBlocks*sizeof(double)),"partial malloc"); check(cudaMalloc(&dsum,sizeof(double)),"sum malloc");
    checksum_partials<<<kChecksumBlocks,kChecksumThreads,0,compute>>>(state,total,dpart); check(cudaGetLastError(),"checksum");
    checksum_fold<<<1,1,0,compute>>>(dpart,kChecksumBlocks,dsum); check(cudaGetLastError(),"checksum fold");
    check(cudaMemcpyAsync(&sum,dsum,sizeof(double),cudaMemcpyDeviceToHost,compute),"sum copy"); check(cudaStreamSynchronize(compute),"done");
    cudaFree(dsum); cudaFree(dpart); stats=ctl.stats(); ctl.reset(); }

    const double layer_updates=(double)layers*(double)tokens;
    std::cout<<std::fixed<<std::setprecision(6)<<"{"
      <<"\"mode\":\""<<tensortransit::to_string(mode)<<"\","<<"\"gpu\":\""<<prop.name<<"\","<<"\"compute_capability\":\""<<prop.major<<"."<<prop.minor<<"\","<<"\"sm_count\":"<<prop.multiProcessorCount<<","<<"\"driver_version\":"<<driver_version<<","<<"\"runtime_version\":"<<runtime_version<<","<<"\"layers\":"<<layers<<","<<"\"state_bytes_per_layer\":"<<state_bytes<<","<<"\"total_state_bytes\":"<<bytes<<","<<"\"tokens\":"<<tokens<<","<<"\"warmup_tokens\":"<<warmup<<","<<"\"prefetch_distance\":"<<distance<<","<<"\"sequences\":"<<sequences<<","<<"\"stream_bytes\":"<<stream_bytes<<","<<"\"hot_set_policy\":\""<<tensortransit::to_string(hot_set_policy)<<"\","<<"\"set_aside_policy\":\""<<tensortransit::to_string(set_aside_policy)<<"\","<<"\"hot_set_model\":\""<<tensortransit::to_string(hot_set_model)<<"\","<<"\"prefetch_schedule\":\""<<tensortransit::to_string(schedule)<<"\","<<"\"prefetch_impl\":\""<<(fused?"fused":"stream")<<"\","<<"\"state_layout\":\""<<(layout==kLinear?"linear":layout==kHeadInterleaved?"head_interleaved":"tile_swapped")<<"\","<<"\"qos\":"<<(qos?"true":"false")<<","<<"\"stream_mode\":\""<<(stream_distinct?"distinct":"reuse")<<"\","<<"\"pre_touch_strategy\":\""<<tensortransit::to_string(pre_touch)<<"\","<<"\"elapsed_ms\":"<<ms<<","<<"\"ms_per_token\":"<<(ms/tokens)<<","<<"\"layer_updates_per_s\":"<<(ms>0.0f?layer_updates/((double)ms/1000.0):0.0)<<","<<"\"l2_bytes\":"<<prop.l2CacheSize<<","<<"\"persisting_l2_max_bytes\":"<<prop.persistingL2CacheMaxSize<<","<<"\"access_policy_max_window_bytes\":"<<prop.accessPolicyMaxWindowSize<<","<<"\"actual_l2_set_aside_bytes\":"<<set_aside<<","<<"\"windows_applied\":"<<stats.windows_applied<<","<<"\"windows_deferred_to_caller\":"<<stats.windows_deferred_to_caller<<","<<"\"hot_set_oversubscribed\":"<<stats.hot_set_oversubscribed<<","<<"\"pre_touch_launches\":"<<stats.pre_touch_launches<<","<<"\"pre_touch_bytes\":"<<stats.pre_touch_bytes<<","<<"\"checksum\":"<<sum<<"}\n";
    cudaEventDestroy(st); cudaEventDestroy(sp); cudaEventDestroy(pf); cudaStreamDestroy(prefetch); cudaStreamDestroy(compute);
    if(weights) cudaFree(weights); if(sink) cudaFree(sink); cudaFree(state); return 0;
 }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
}
