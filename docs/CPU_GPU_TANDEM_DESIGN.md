# kindi CPU+GPU Tandem Inference Design

**Status**: Design Phase
**Author**: Systems Research
**Date**: 2026-11-07
**Target**: kindi v0.5.0+

---

## Executive Summary

This document designs a **CPU+GPU tandem inference architecture** for kindi, inspired by [tandem.cpp](https://github.com/ewhacc/tandem.cpp)'s approach for MoE models. While tandem.cpp targets Mixture-of-Experts models with expert-level granularity, kindi runs **dense and MoE models** with different quantization schemes where the natural unit is the **layer**.

### Kindi Model Zoo (v0.4.0)

| Model Family | Variants | Quantization | Type | VRAM Target |
|--------------|----------|--------------|------|-------------|
| **Ternary-Bonsai-2** | 2B, 4B, 9B, 27B | PTQ1_0 (ternary, ~1.58 bit) | Dense | 1 GB (2B) → 8 GB (27B) |
| **Nemotron 3.5 Lightning** | 30B (3B active) | Various | **MoE** (8 experts, top-2) | 12–24 GB |
| **Nemotron 3.5 Bolt / Bolt-E4B** | 8B total / 4B effective | Various | Dense | < 6 GB (4B eff.) |
| **AH-Nanbeige4.3** | 33B, 3B | IQ3_XXS (3-bit) | Dense | 10 GB (33B), ~2 GB (3B) |

**Key distinction**: Only **Ternary-Bonsai-2** uses ternary (1.58-bit) weights. Nemotron and Nanbeige use standard quantization (3-bit, 4-bit, etc.). The tandem design must handle both dense and MoE layer types.

**Goal**: Run models larger than VRAM (e.g., 27B on 8GB, 33B on 10GB) with minimal latency penalty by:
- Keeping only active layers on GPU
- Offloading inactive layers + KV cache to CPU RAM
- Streaming weights asynchronously (overlap compute + transfer)
- Prefetching next layer during current layer execution

---

## 1. Architecture Overview

### 1.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KINDI TANDEM ENGINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │   PROMPT     │     │   DECODE     │     │   SPECULATIVE│               │
│   │   PIPELINE   │────▶│   PIPELINE   │────▶│   PIPELINE   │               │
│   └──────────────┘     └──────────────┘     └──────────────┘               │
│         │                    │                    │                        │
│         ▼                    ▼                    ▼                        │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                    LAYER SCHEDULER                                │    │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐  │    │
│   │  │ Layer 0 │  │ Layer 1 │  │ Layer 2 │  │ ...     │  │ Layer N│  │    │
│   │  │ (GPU)   │  │ (CPU→GPU)│ │ (CPU)   │  │         │  │ (CPU)  │  │    │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └────────┘  │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│         │                    │                    │                        │
│         ▼                    ▼                    ▼                        │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │  GPU MEMORY  │◀───▶│  CPU RAM     │◀───▶│  DISK (mmap) │               │
│   │  (Hot Layers,│     │  (Warm Layers,│     │  (Cold Layers│               │
│   │   KV Cache)  │     │   KV Overflow)│     │   Weights)   │               │
│   └──────────────┘     └──────────────┘     └──────────────┘               │
│         ▲                    ▲                    ▲                        │
│         │                    │                    │                        │
│   ┌─────┴─────┐      ┌──────┴──────┐      ┌─────┴─────┐                   │
│   │ GPU       │      │ CPU WORKER  │      │ STREAM    │                   │
│   │ KERNELS   │      │ THREAD      │      │ PREFETCH  │                   │
│   └───────────┘      └─────────────┘      └───────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Concepts (Adapted from tandem.cpp)

| tandem.cpp (MoE) | kindi Tandem (Dense) |
|------------------|----------------------|
| Expert cache (LRU) | **Layer cache** (LRU, 1-2 layers hot) |
| Doorbell (async CPU-GPU) | **Doorbell per layer** (signal CPU to prep next layer) |
| Lookahead (router predicts) | **Lookahead** (current layer streams next layer weights) |
| Prefill share (large batch) | **Chunked prefill** (batch prompt tokens, overlap layers) |
| Hybrid CPU/GPU split | **Hybrid layer split** (attention on GPU, FFN on CPU optional) |

---

## 2. Integration Points in kindi Codebase

---

## 3. Detailed Design

### 3.1 Layer Cache (GPU Hot Storage)

```python
# kindi/tandem.py
class LayerCache:
    """LRU cache of layer weights on GPU. Like tandem.cpp's expert cache but per-layer."""

    def __init__(self, engine, max_layers_gpu: int = 2, max_mib: int = None):
        self.engine = engine
        self.max_layers = max_layers_gpu
        self.max_bytes = max_mib * 1024 * 1024 if max_mib else None
        self.layers_on_gpu = {}           # layer_idx -> {weight_name: device_ptr}
        self.lru = []                     # [layer_idx] most recent last
        self.pinned_cpu_buffers = {}      # layer_idx -> {weight_name: host_ptr}
        self.stream = cp.cuda.Stream()    # Dedicated copy stream
        self.copy_events = {}             # layer_idx -> event

    def ensure_layer(self, layer_idx: int) -> bool:
        """Ensure layer is on GPU. Returns True if already resident."""
        if layer_idx in self.layers_on_gpu:
            self._touch(layer_idx)
            return True
        return self._load_layer(layer_idx)

    def _load_layer(self, layer_idx: int) -> bool:
        """Async load layer from CPU pinned memory to GPU."""
        if len(self.layers_on_gpu) >= self.max_layers:
            victim = self.lru.pop(0)
            self._evict(victim)

        # Launch async copy on dedicated stream
        ev = self.stream.record()
        for name, cpu_ptr in self.pinned_cpu_buffers[layer_idx].items():
            gpu_ptr = self.engine.layer_ptrs[layer_idx][name]
            cp.cuda.runtime.memcpyAsync(gpu_ptr, cpu_ptr, size, self.stream.ptr)
        self.copy_events[layer_idx] = ev
        self.layers_on_gpu[layer_idx] = self.engine.layer_ptrs[layer_idx]
        self.lru.append(layer_idx)
        return False  # Was not resident

    def wait_layer(self, layer_idx: int):
        """Block until layer copy completes."""
        if layer_idx in self.copy_events:
            self.copy_events[layer_idx].synchronize()
            del self.copy_events[layer_idx]

    def prefetch_next(self, current_layer: int):
        """Launch async load of next layer while current executes."""
        next_layer = current_layer + 1
        if next_layer < self.engine.num_layers and next_layer not in self.layers_on_gpu:
            self._load_layer(next_layer)
```

### 3.2 Doorbell Mode (Async CPU-GPU Handoff)

Adapted from tandem.cpp's `GGML_OP_MOE_DOORBELL`:

```python
# kindi/tandem.py
class LayerDoorbell:
    """Async signal from GPU to CPU worker: 'prepare layer N'."""

    def __init__(self, num_layers: int):
        # Pinned host memory visible to both CPU and GPU
        self.flags = cp.cuda.PinnedMemory(num_layers * 4)  # uint32 per layer
        self.flag_array = np.frombuffer(self.flags, dtype=np.uint32)
        self.flag_array[:] = 0
        self.cpu_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.cpu_thread.start()
        self.job_queue = queue.Queue()
        self.shutdown = False

    def ring(self, layer_idx: int):
        """GPU calls this (via kernel) to request layer prep."""
        # Atomic increment - GPU writes, CPU reads
        self.flag_array[layer_idx] += 1

    def _worker_loop(self):
        while not self.shutdown:
            for i in range(self.num_layers):
                if self.flag_array[i] > self.ack_array[i]:
                    self._prepare_layer(i)
                    self.ack_array[i] = self.flag_array[i]
            time.sleep(0.0001)  # 100µs poll

    def _prepare_layer(self, layer_idx: int):
        """CPU-side: decompress, transpose, stage in pinned memory."""
        # 1. Load from mmap/disk if not in RAM
        # 2. Repack to GPU layout (perm8, 5-trit, etc.)
        # 3. Copy to pinned CPU buffer for fast GPU upload
        pass
```

**CUDA Kernel for Doorbell** (`tandem_kernels.cu`):
```cuda
__global__ void k_doorbell_signal(uint32_t *flags, int layer_idx) {
    atomicAdd(&flags[layer_idx], 1);
}

// Host launcher
void tandem_doorbell_ring(cudaStream_t stream, uint32_t *flags, int layer_idx) {
    k_doorbell_signal<<<1, 1, 0, stream>>>(flags, layer_idx);
}
```
### 2.1 Core Files to Modify

```
kindi/
├── engine.py          # Main GPU engine - ADD layer scheduler, weight streaming
├── cpu.py             # CPU engine - ADD async worker thread, doorbell handler
├── cache.py           # Prompt cache - ADD CPU KV overflow, checkpoint streaming
├── tp.py              # Tensor parallel - ADD cross-card layer offload
├── spec.py            # Speculative decoding - ADD draft model on CPU
├── config.py          # Configuration - ADD tandem env vars
├── __main__.py        # CLI - ADD tandem bench/serve flags
├── layout.py          # Weight layout - ADD streaming-friendly layout
├── hostram.py         # NEW: CPU memory management (pinned, mmap)
├── tandem.py          # NEW: Tandem engine orchestration
└── tandem_kernels.cu  # NEW: CUDA kernels for async copy, doorbell
```

### 3.3 Weight Streaming Layout

Current kindi layout scatters weights across files. For streaming, we need **layer-contiguous** layout:

```
kindi_pack/
├── kindi_pack.json           # Manifest
├── layer_00.bin              # All layer 0 weights (contiguous)
├── layer_01.bin
├── ...
├── layer_23.bin
├── shared/
│   ├── token_embd.bin        # Embedding (always pinned)
│   ├── output_norm.bin
│   └── head.bin
└── metadata/
    ├── layer_offsets.json    # {layer_idx: {weight_name: offset, size}}
    └── streaming_order.json  # Optimal prefetch order
```

**Loading Strategy**:
```python
# kindi/hostram.py
class StreamingWeightLoader:
    def __init__(self, pack_path: str, pin_memory: bool = True):
        self.pack_path = pack_path
        self.pin_memory = pin_memory
        self.layer_files = {}      # layer_idx -> mmap handle
        self.pinned_buffers = {}   # layer_idx -> {weight_name: pinned_ptr}
        self.metadata = json.load(open(pack_path / "metadata/layer_offsets.json"))

    def ensure_pinned(self, layer_idx: int):
        """Load layer into pinned host memory (once)."""
        if layer_idx in self.pinned_buffers:
            return

        layer_file = self.layer_files[layer_idx]
        buffers = {}
        for name, (offset, size) in self.metadata[str(layer_idx)].items():
            if self.pin_memory:
                ptr = cp.cuda.alloc_pinned_memory(size)
            else:
                ptr = np.empty(size, dtype=np.uint8)
            # Copy from mmap to pinned
            np.copyto(ptr, layer_file[offset:offset+size])
            buffers[name] = ptr
        self.pinned_buffers[layer_idx] = buffers

    def evict_pinned(self, layer_idx: int):
        """Free pinned memory for layer."""
        if layer_idx in self.pinned_buffers:
            for ptr in self.pinned_buffers[layer_idx].values():
                if self.pin_memory:
                    cp.cuda.free(ptr.ptr)
            del self.pinned_buffers[layer_idx]
```

### 3.4 KV Cache Offloading (CPU Overflow)

Extend `cache.py` to spill KV cache to CPU RAM when GPU fills:

```python
# kindi/cache.py additions
class KVOffloadManager:
    """Manages KV cache across GPU (hot) and CPU RAM (warm)."""

    def __init__(self, engine, max_gpu_layers: int = None, cpu_budget_mb: int = 0):
        self.engine = engine
        self.gpu_layers = max_gpu_layers or engine.num_layers
        self.cpu_budget = cpu_budget_mb * 1024 * 1024
        self.cpu_kv = {}           # layer_idx -> (K_cpu, V_cpu) in pinned memory
        self.gpu_resident = set()  # layers with KV on GPU
        self.lru = []              # GPU KV LRU

    def ensure_kv_on_gpu(self, layer_idx: int, seq_len: int):
        """Bring KV for layer to GPU if needed."""
        if layer_idx in self.gpu_resident:
            self._touch_kv(layer_idx)
            return

        # Evict if needed
        while self._gpu_kv_bytes() + self._layer_kv_bytes(layer_idx, seq_len) > self.gpu_kv_budget:
            victim = self.lru.pop(0)
            self._spill_kv_to_cpu(victim)

        if layer_idx in self.cpu_kv:
            self._restore_kv_from_cpu(layer_idx, seq_len)
        else:
            # Fresh layer - allocate on GPU
            self._alloc_kv_on_gpu(layer_idx, seq_len)
        self.gpu_resident.add(layer_idx)
        self.lru.append(layer_idx)

    def _spill_kv_to_cpu(self, layer_idx: int):
        """Async copy KV from GPU to pinned CPU memory."""
        k_gpu, v_gpu = self.engine.k_cache[layer_idx], self.engine.v_cache[layer_idx]
        k_cpu = cp.cuda.alloc_pinned_memory(k_gpu.nbytes)
        v_cpu = cp.cuda.alloc_pinned_memory(v_gpu.nbytes)
        stream = cp.cuda.Stream()
        k_gpu.get(stream=stream, out=k_cpu)
        v_gpu.get(stream=stream, out=v_cpu)
        stream.synchronize()
        self.cpu_kv[layer_idx] = (k_cpu, v_cpu)
        self.gpu_resident.remove(layer_idx)
        # Free GPU memory
        self.engine.k_cache[layer_idx] = None
        self.engine.v_cache[layer_idx] = None
```

### 3.5 Lookahead Prefetching

While layer `i` computes on GPU, stream layer `i+1` weights:

```python
# kindi/tandem.py
class LookaheadPrefetcher:
    """Overlap weight streaming with computation."""

    def __init__(self, engine, cache: LayerCache, loader: StreamingWeightLoader):
        self.engine = engine
        self.cache = cache
        self.loader = loader
        self.prefetch_stream = cp.cuda.Stream()
        self.active_prefetch = None

    def start_layer(self, layer_idx: int):
        """Called at start of layer execution."""
        # 1. Wait for any pending prefetch to complete
        if self.active_prefetch is not None:
            self.cache.wait_layer(self.active_prefetch)
            self.active_prefetch = None

        # 2. Ensure current layer is resident
        self.cache.wait_layer(layer_idx)

        # 3. Launch prefetch for next layer
        next_layer = layer_idx + 1
        if next_layer < self.engine.num_layers:
            self.loader.ensure_pinned(next_layer)
            self.cache._load_layer(next_layer)
            self.active_prefetch = next_layer

    def end_layer(self, layer_idx: int):
        """Called at end of layer execution."""
        # Prefetch continues in background
        pass
```

### 3.6 Hybrid CPU/GPU Layer Execution (Optional)

For very memory-constrained GPUs (K420 1GB), split layer computation:

```
Layer i on GPU:     Attention (QKV, scores, O) + RMSNorm
Layer i on CPU:     FFN (Gate/Up/Down) + GDN (if applicable)
```

This requires:
- GPU computes attention → residual in pinned memory
- CPU worker picks up residual → computes FFN/GDN → writes back to pinned memory
- GPU reads result → continues to next layer

**Doorbell integration**: GPU kernels signal CPU worker via doorbell when attention done.
### 2.2 Existing Infrastructure to Leverage

| Component | Current Use | Tandem Extension |
|-----------|-------------|------------------|
| `engine.Engine` | Single-GPU dense inference | Base class for `TandemEngine` |
| `cache.Cache` | Prompt prefix reuse | CPU KV overflow, checkpoint streaming |
| `tp.TPEngine` | 2-GPU tensor parallel | Multi-GPU layer distribution + CPU offload |
---

## 4. Configuration System

### 4.1 Environment Variables (tandem.cpp style)

```bash
# Layer cache
KINDI_TANDEM_LAYERS_GPU=2          # Max layers resident on GPU (default: auto)
KINDI_TANDEM_CACHE_MIB=auto        # GPU cache budget in MiB (auto = free VRAM - margin)
KINDI_TANDEM_CACHE_MARGIN_MIB=1024 # Safety margin for KV, compute buffers

# CPU offload
KINDI_TANDEM_CPU_LAYERS=0          # Force first N layers on CPU (0 = auto)
KINDI_TANDEM_CPU_FFN=0             # Run FFN on CPU (hybrid mode)

# Prefetching
KINDI_TANDEM_PREFETCH=1            # Enable lookahead prefetch (default: 1)
KINDI_TANDEM_PREFETCH_LAYERS=1     # How many layers ahead to prefetch

# KV offload
KINDI_TANDEM_KV_CPU_MIB=0          # CPU RAM budget for KV overflow (0 = 20% RAM)
KINDI_TANDEM_KV_PINNED=1           # Use pinned memory for CPU KV

# Weight streaming
KINDI_TANDEM_STREAM_WEIGHTS=1      # Stream from disk/mmap (default: 1)
KINDI_TANDEM_PIN_WEIGHTS=1         # Keep weights in pinned host RAM

# Doorbell / async
KINDI_TANDEM_DOORBELL=1            # Enable async CPU-GPU handoff
KINDI_TANDEM_CPU_THREADS=0         # CPU worker threads (0 = auto)

# Profiling
KINDI_TANDEM_PROFILE=0             # Enable timing logs
KINDI_TANDEM_STATS=0               # Print cache stats at exit
```

### 4.2 Config File Keys (`kindi.conf`)

```ini
[kindi]
# ... existing keys ...

# Tandem CPU+GPU offload
tandem_layers_gpu = auto        # 1, 2, 3, or "auto"
tandem_cache_mib = auto         # GPU cache budget MiB
tandem_cache_margin_mib = 1024  # Safety margin
tandem_cpu_layers = 0           # Force CPU layers
tandem_cpu_ffn = 0              # Hybrid FFN on CPU
tandem_prefetch = 1             # Lookahead prefetch
tandem_prefetch_layers = 1      # Layers ahead
tandem_kv_cpu_mib = 0           # CPU KV budget (0 = 20% RAM)
tandem_kv_pinned = 1            # Pinned CPU KV
tandem_stream_weights = 1       # Stream from disk
tandem_pin_weights = 1          # Pinned host weights
tandem_doorbell = 1             # Async handoff
tandem_cpu_threads = 0          # CPU worker threads
tandem_profile = 0              # Timing logs
tandem_stats = 0                # Cache stats
```

### 4.3 Precedence
Built-in defaults < Config file < Environment (`KINDI_TANDEM_*`) < CLI flags
| `spec.install` | Tensor-core speculative | CPU draft model, async verify |
| `hostram.pinned` | Embedding in pinned memory | All CPU-resident weights in pinned memory |
| `layout.repack` | Weight layout for GEMV | Streaming layout (layer-contiguous) |
---

## 5. Prototype Implementation Plan

### Phase 1: Foundation (Week 1-2)
**Goal**: Weight streaming + layer cache + basic offload

| Task | File | Description |
|------|------|-------------|
| 1.1 | `hostram.py` | NEW: Pinned memory allocator, mmap weight loader |
| 1.2 | `layout.py` | Add `repack_streaming()` for layer-contiguous layout |
| 1.3 | `convert.py` | Emit streaming layout (layer_XX.bin + metadata) |
| 1.4 | `tandem.py` | NEW: `LayerCache`, `StreamingWeightLoader` classes |
| 1.5 | `engine.py` | Add `TandemEngine` subclass with layer-by-layer execution |
| 1.6 | `config.py` | Add tandem config keys, env var parsing |
| 1.7 | `__main__.py` | Add `--tandem` flag to serve/chat/bench |

**Test**: 27B model on RTX 3080 (10GB) - should fit with 2 layers on GPU

### Phase 2: Async Pipeline (Week 2-3)
**Goal**: Doorbell + prefetching + overlap

| Task | File | Description |
|------|------|-------------|
| 2.1 | `tandem_kernels.cu` | NEW: `k_doorbell_signal`, async copy kernels |
| 2.2 | `tandem.py` | `LayerDoorbell`, `LookaheadPrefetcher` classes |
| 2.3 | `engine.py` | Integrate doorbell in layer loop, dual-stream execution |
| 2.4 | `cpu.py` | Add CPU worker thread for doorbell handling |

**Test**: Measure decode tok/s with/without prefetch on 27B/RTX 3080

### Phase 3: KV Offload + Prompt Cache Integration (Week 3-4)
**Goal**: Long context on small VRAM

| Task | File | Description |
|------|------|-------------|
| 3.1 | `cache.py` | `KVOffloadManager`, integrate with `Cache.resume()` |
| 3.2 | `tandem.py` | Coordinate layer cache + KV offload |
| 3.3 | `engine.py` | KV allocation from offload manager |

**Test**: 32K context on 8GB GPU (KV offload to CPU RAM)

### Phase 4: Hybrid CPU/GPU Layer Split (Week 4-5)
**Goal**: Run on K420 (1GB), K1200 (4GB)

| Task | File | Description |
|------|------|-------------|
| 4.1 | `tandem.py` | `HybridLayerExecutor` - split attention/FFN |
| 4.2 | `cpu.py` | Add FFN/GDN kernels for CPU execution |
| 4.3 | `tandem_kernels.cu` | Residual exchange kernels (GPU↔CPU) |
| 4.4 | `engine.py` | Hybrid execution path |

**Test**: 9B on K420 (1GB), 27B on K1200 (4GB)

### Phase 5: Speculative + Multi-GPU (Week 5-6)
**Goal**: Full feature parity

| Task | File | Description |
|------|------|-------------|
| 5.1 | `spec.py` | CPU draft model, async verify with tandem |
| 5.2 | `tp.py` | Tensor parallel + layer offload combined |
| 5.3 | `bench.py` | Tandem-specific benchmarks |
---

## 6. Target Benchmarks

### 6.1 Hardware Targets

| GPU | VRAM | Target Models | Expected Decode (tok/s) |
|-----|------|---------------|------------------------|
| K420 | 1 GB | Bonsai 2B, 4B | 2B: ~15, 4B: ~8 |
| K1200 | 4 GB | Bonsai 9B, 27B (2-card TP) | 9B: ~12, 27B: ~6 |
| GTX 1080 | 8 GB | Bonsai 27B, Nemotron 4B | 27B: ~8, 4B: ~25 |
| RTX 3080 | 10 GB | Bonsai 27B, Nemotron 4B | 27B: ~15, 4B: ~40 |
| RTX 4090 | 24 GB | All (full GPU) | Baseline (no tandem needed) |

### 6.2 Success Criteria

| Metric | Target |
|--------|--------|
| **27B on RTX 3080 (10GB)** | ≥ 12 tok/s decode (vs ~0 OOM baseline) |
| **9B on K1200 (4GB)** | ≥ 10 tok/s decode |
| **4B on K420 (1GB)** | ≥ 8 tok/s decode |
| **32K context on 8GB** | Works without OOM |
| **Prefetch speedup** | ≥ 1.3× vs no-prefetch |
| **Quality** | Token-exact match with full-GPU greedy |

### 6.3 Profiling Metrics to Collect

```python
# tandem.py - built-in profiler
class TandemProfiler:
    def __init__(self):
        self.layer_load_us = []      # Weight load time per layer
        self.layer_compute_us = []   # Compute time per layer
        self.prefetch_overlap_us = [] # Overlap achieved
        self.cache_hits = 0
        self.cache_misses = 0
        self.kv_spills = 0
        self.kv_restores = 0
        self.doorbell_signals = 0
        self.cpu_wait_us = 0
```

---

## 7. Risk Analysis & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PCIe bandwidth bottleneck | High | Medium | Compress weights (quantize to 4-bit for transfer), overlap with compute |
| CPU-GPU sync latency | Medium | High | Doorbell + double-buffering, measure and tune |
| Pinned memory exhaustion | Medium | Medium | Configurable limits, fall back to pageable |
| Quality regression | Low | Critical | Token-exact verification at each phase |
| Complexity explosion | High | High | Phase gates, feature flags, comprehensive tests |

---

## 8. File Structure Summary

```
kindi/
├── hostram.py              # NEW: Pinned memory, mmap, streaming loader
├── tandem.py               # NEW: LayerCache, Doorbell, Prefetcher, TandemEngine
├── tandem_kernels.cu       # NEW: Doorbell signal, async copy kernels
├── layout.py               # MOD: streaming layout (layer-contiguous)
├── convert.py              # MOD: emit streaming layout
├── engine.py               # MOD: TandemEngine subclass
├── cpu.py                  # MOD: CPU worker thread, doorbell handler
├── cache.py                # MOD: KVOffloadManager
├── config.py               # MOD: tandem config keys
├── __main__.py             # MOD: --tandem flags
├── spec.py                 # MOD: CPU draft model
├── tp.py                   # MOD: TP + offload combined
└── bench.py                # MOD: tandem benchmarks
```

---

## 9. Next Steps

1. **Immediate**: Create `hostram.py` with pinned memory allocator and mmap loader
2. **Week 1**: Implement streaming layout in `convert.py` + `layout.py`
3. **Week 1**: Build `LayerCache` and `TandemEngine` skeleton
4. **Week 2**: Add doorbell CUDA kernel and CPU worker
5. **Week 2**: Integrate lookahead prefetcher
6. **Week 3**: KV offload + prompt cache integration
7. **Week 4**: Hybrid CPU/GPU layer split for K420/K1200
8. **Week 5**: Speculative + multi-GPU tandem
9. **Week 6**: Benchmarks, tuning, documentation

---

## Appendix: Tandem.cpp Key Structures Reference

### `llama_moe_cache::impl` (simplified)
```cpp
struct impl {
    struct group {                    // GPU device group
        vector<int32_t> layers;       // Layers assigned to this GPU
        moe_cache_lru lru;            // LRU cache per layer
        ggml_backend_t backend;       // CUDA backend
    };
    vector<group> groups;

    // CPU worker thread
    thread worker;
    atomic<int> job_layer;
    void worker_loop();               // Computes missed experts

    // Doorbell
    thread db_thread;
    struct ggml_moe_db *db;           // Control block in pinned memory
    void db_loop();                   // Handles doorbell requests

    // Lookahead prefetch
    thread la_thread;
    ggml_backend_t la_backend;        // Second backend for prefetch
    void la_loop();                   // Prefetches predicted experts

    // Hybrid CPU/GPU split
    double hybrid_share;              // Fraction to CPU
    void share_update(wait_us, idle_us); // Adaptive split
};
```

### Key Env Vars (tandem.cpp)
```
LLAMA_MOE_CACHE_MIB=auto        # GPU cache size
LLAMA_MOE_CACHE_MARGIN_MIB=1024 # Safety margin
LLAMA_MOE_CPU=0.2               # Fixed CPU share (0-1)
LLAMA_MOE_PREFILL_CPU=1         # CPU for prefill
LLAMA_MOE_LOOKA=1               # Lookahead enable
LLAMA_MOE_LOOKA_K=4             # Top-K experts to predict
LLAMA_MOE_LOOKA_BUDGET=8        # Max experts to prefetch
LLAMA_MOE_PIN_EXPS=1            # Pin experts in RAM
```

---

*End of Design Document*
