#include "recurlocal/planner.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace recurlocal;
int main() {
    DeviceCaps caps{96ull*1024*1024, 64ull*1024*1024, 32ull*1024*1024};
    PlannerConfig cfg; cfg.mode=LocalityMode::Combined; cfg.persisting_budget_fraction=0.5; cfg.hit_ratio=0.8;
    LocalityPlanner p(caps,cfg);
    assert(p.recommended_l2_set_aside()==32ull*1024*1024);
    auto x=p.plan_for_layer(3ull*1024*1024,true);
    assert(x.use_persisting_window && x.prefetch_next && x.hot_window_bytes==3ull*1024*1024);
    assert(std::abs(x.hit_ratio-0.8)<1e-9);
    PlannerConfig b=cfg; b.mode=LocalityMode::Baseline; LocalityPlanner pb(caps,b);
    auto y=pb.plan_for_layer(3ull*1024*1024,true); assert(!y.use_persisting_window && !y.prefetch_next);
    PlannerConfig c=cfg; c.max_hot_window_bytes=1024*1024; LocalityPlanner pc(caps,c);
    assert(pc.plan_for_layer(3ull*1024*1024,true).hot_window_bytes==1024ull*1024);
    bool threw=false; try { PlannerConfig bad=cfg; bad.hit_ratio=1.1; LocalityPlanner nope(caps,bad); } catch(const std::invalid_argument&) { threw=true; }
    assert(threw);
    std::cout << "planner tests passed\n";
}
