# CUDA Optimization Benchmark Template

Clone-and-go harness for benchmarking CPU and GPU kernel implementations side by side, with per-phase timing (H2D / kernel / D2H) and correctness comparison against a CPU reference.

---

## Structure

```
.
├── include/
│   ├── benchmark_utils.h   # timing harness, Buffers RAII, compare_outputs   [edit if need different memory layout]
│   └── func_interface.h    # Params struct + function signature              [edit]
├── src/
│   ├── benchmark.cu        # orchestration, reporting                        [edit]
│   ├── func_cpu.cpp        # CPU reference implementation                    [edit]
│   └── func_naive.cu       # naive GPU implementation                        [edit]
├── build/                  # object files (generated)
├── bin/                    # executable (generated)
└── Makefile
```

---

## Checklist when starting a new algorithm

**`include/func_interface.h`**
- [ ] Rename `XXXfunc` to your algorithm (e.g. `conv`, `reduce`, `scan`)
- [ ] Fill in `Params` fields with the problem dimensions
- [ ] Implement `input_size()` and `output_size()` — these drive all allocations in `Buffers`

**`src/benchmark.cu`**
- [ ] Add a `namespace` block per implementation tier you want to benchmark
- [ ] Set `Params` fields to your target problem size
- [ ] Fill in the `printf("Function: ")` label

**`src/func_cpu.cpp`**
- [ ] Implement the CPU reference — this is the correctness baseline everything else is compared against

**`src/func_naive.cu`**
- [ ] Implement the baseline GPU kernel
- [ ] Match the signature `void XXXfunc(const float*, float*, const Params&)`
- [ ] No `cudaMemcpy` inside — transfers are timed separately by the harness

**`include/benchmark_utils.h`**
- [ ] Implement `calculate_gflops()` — fill in `ops` with the actual FLOP count for your algorithm (currently returns 0)

**`Makefile`**
- [ ] Set `ARCH` to match your GPU (default is `sm_89` / RTX 4060 Ti)
- [ ] Add new implementation `.cu` files to `CUDA_SRC`
- [ ] Rename `TARGET` if desired

---

## Build

```bash
make                  # build for default arch (sm_89)
make ARCH=sm_90       # H100
make ARCH=sm_89       # Ada / RTX 40xx
make ARCH=sm_86       # Ampere / RTX 30xx
make ARCH=sm_80       # A100
```

### Adding a new implementation

1. Create `src/func_optimised.cu` with the matching signature
2. Add it to `CUDA_SRC` in the Makefile:
   ```makefile
   CUDA_SRC = $(SRC_DIR)/func_naive.cu \
              $(SRC_DIR)/func_optimised.cu
   ```
3. Add a namespace declaration in `src/benchmark.cu`:
   ```cpp
   namespace optimised {
   void XXXfunc(const float*, float*, const Params&);
   }
   ```

---

## Run

```bash
make run
```

---

## Profiling

| Target | Tool | Use when |
|---|---|---|
| `make profile` | ncu --set full | Full metric deep-dive |
| `make profile-quick` | ncu --set basic | Quick sanity check |
| `make profile-kernel` | ncu, regex filter | Isolating a specific kernel |
| `make profile-sys` | nsys | Timeline, H2D/D2H overlap, CPU/GPU interaction |

```bash
make profile          # saves profile_ncu.ncu-rep
make profile-sys      # saves profile_nsys.nsys-rep

# Open results
ncu-ui  profile_ncu.ncu-rep
nsys-ui profile_nsys.nsys-rep
```

`profile-kernel` targets any kernel whose name matches `func_.*` — rename the
regex in the Makefile if your kernel is named differently.

---

## Output format

```
Function: naive convolution
Output Total FLOPs: 18.87 GFLOP

Comparing naive output to CPU reference...
  Max absolute difference: 1.24e-06
  Max relative error:      3.17e-05

Performance:
Baseline                 :    4.821 ms
Naive                    :    1.203 ms  (  4.01x speedup)
  └─ H2D: 0.312 ms, Kernel: 0.744 ms, D2H: 0.147 ms
  CPU:    3.92 GFLOPS
  Naive:  25.36 GFLOPS
```

The H2D / kernel / D2H breakdown is the most useful column when presenting
results — it immediately shows whether the bottleneck is the kernel or the
transfer, and whether an async pipeline would help.

---

## Utility targets

```bash
make info       # CUDA version, GPU name, driver, compute capability, build config
make clean      # remove build/ and bin/
make cleanall   # clean + delete all .ncu-rep / .nsys-rep profile files
make help       # print all available targets
```
