# Frontier model

RecurLocal is intended for self-directed optimization. Contributors profile current `main`, find a bottleneck, and demonstrate an improvement.

Primary release metrics should be real-model decode throughput, concurrency throughput, recurrent-state HBM traffic, recurrent-kernel time, and energy.

A future release headline may look like this **only after measurement**:

> RecurLocal v0.x: +8.4% Qwen3.8 decode on one RTX 5090, with 3.1× lower recurrent-state DRAM traffic and identical output.

Never publish invented values.

Secondary metrics such as L2 hit rate explain why a change works but do not replace real end-to-end serving metrics.
