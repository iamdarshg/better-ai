# Better AI - Unified Comprehensive TODO

**Last Updated:** February 14, 2026  
**Priority:** Critical improvements for agentic coding and scientific tasks on edge systems

---

## 🚨 CRITICAL ISSUES (Immediate Action Required)

### 1. Dataset System Migration & Cleanup (Priority: CRITICAL)
**Estimated Effort:** 3-5 days

#### Problems Identified:
- Old dataset-specific classes exist alongside new unified datasets.yml system
- Duplicate implementations causing confusion and maintenance burden
- `CodeDataset`, `MixedCodeDataset`, `ExpertAwareDataset` are now redundant
- Inconsistent dataset loading patterns across codebase

#### Action Items:
- [ ] **Remove obsolete dataset classes:**
  - [ ] Delete `better_ai/data/datasets/code_dataset.py` (replaced by unified_dataloader.py + datasets.yml)
  - [ ] Delete `better_ai/data/datasets/mixed_code_dataset.py` (mixing now handled by curriculum)
  - [ ] Delete `better_ai/data/datasets/expert_aware_dataset.py` (expert routing integrated in model)
  - [ ] Keep only `rolling_window_dataset.py` for streaming implementation
- [ ] **Update all imports:**
  - [ ] Search codebase for `from better_ai.data.datasets import CodeDataset` and remove
  - [ ] Replace with `UnifiedDataLoader` from `better_ai/data/unified_dataloader.py`
  - [ ] Verify no broken imports remain
- [ ] **Consolidate dataset loading:**
  - [ ] Ensure all dataset references use datasets.yml configuration
  - [ ] Validate all 50+ datasets defined in datasets.yml load correctly
  - [ ] Add validation script to test dataset loading on startup
- [ ] **Update documentation:**
  - [ ] Update README.md with unified dataset system usage
  - [ ] Add migration guide for old dataset class users
  - [ ] Document datasets.yml schema and curriculum integration

---

### 2. Stub Implementation Completion (Priority: CRITICAL)
**Estimated Effort:** 2-3 weeks

#### 2.1 Fault Localization & Patch Generation
**File:** `better_ai/training/fault_localization.py`  
**Current State:** Minimal stub with hardcoded examples  
**Estimated Effort:** 1 week

**Problems:**
- `FaultLocalizer.localize_fault()` returns hardcoded suspiciousness scores
- No actual static analysis or trace parsing
- `PatchGenerator.generate_patches()` returns template strings, not real patches
- No integration with actual code execution or testing
- Missing AST analysis for Python code
- No support for C/Rust fault localization

**Action Items:**
- [ ] Implement real AST-based fault localization:
  - [ ] Parse error traces to extract file/line/column information
  - [ ] Map stack traces to source code locations
  - [ ] Calculate suspiciousness scores using spectrum-based techniques (Tarantula, Ochiai)
  - [ ] Support Python, C, and Rust error trace formats
- [ ] Implement real patch generation:
  - [ ] Generate multiple patch candidates using model inference
  - [ ] Use LLM with scratchpad reasoning for patch synthesis
  - [ ] Validate patches with syntax checking (AST validation)
  - [ ] Score patches based on likelihood and test coverage
- [ ] Add test execution environment:
  - [ ] Integrate with code execution sandbox
  - [ ] Run unit tests to validate patches
  - [ ] Support pytest, unittest, cargo test, CTest
- [ ] Implement reward function:
  - [ ] Compute patch quality score based on test pass rate
  - [ ] Add localization accuracy metric (fault in top-N ranked lines)
  - [ ] Combine into unified repair reward for RLHF

#### 2.2 Reasoning Diversity Metrics
**File:** `better_ai/training/diversity_metrics.py`  
**Current State:** Basic n-gram and cosine similarity only  
**Estimated Effort:** 4-5 days

**Problems:**
- Only measures surface-level n-gram diversity
- Embedding diversity uses simple cosine similarity
- No semantic diversity measurement
- Missing solution approach classification
- Not integrated into RLHF training loop

**Action Items:**
- [ ] Implement semantic diversity metrics:
  - [ ] Add semantic role labeling to identify reasoning patterns
  - [ ] Measure diversity of intermediate reasoning steps
  - [ ] Detect duplicate logical paths with different wording
  - [ ] Add tree edit distance for reasoning tree structures
- [ ] Implement solution approach classification:
  - [ ] Cluster reasoning trajectories by approach (brute-force, greedy, DP, etc.)
  - [ ] Reward groups with diverse approach distribution
  - [ ] Penalize convergence to single approach
- [ ] Integrate with GRPO/ARPO training:
  - [ ] Add diversity bonus to group advantage estimation
  - [ ] Balance diversity vs. correctness in reward function
  - [ ] Track diversity metrics over training
- [ ] Add visualization:
  - [ ] Plot diversity evolution during training
  - [ ] Visualize reasoning trajectory clusters
  - [ ] Generate diversity reports per checkpoint

#### 2.3 Model Pruning
**File:** `better_ai/training/pruning.py`  
**Current State:** Only zeros weights, doesn't remove them  
**Estimated Effort:** 3-4 days

**Problems:**
- `prune_expert_widths()` only zeros weights, doesn't reduce model size
- No actual parameter removal or architecture shrinking
- No support for structured pruning
- Missing pruning-aware fine-tuning
- No pruning metrics or analysis

**Action Items:**
- [ ] Implement true weight removal:
  - [ ] Reconstruct Linear layers with reduced dimensions after pruning
  - [ ] Update bias vectors to match pruned weight dimensions
  - [ ] Verify model forward pass works after pruning
- [ ] Add structured pruning:
  - [ ] Support channel/neuron pruning (entire columns/rows)
  - [ ] Implement head pruning for attention layers
  - [ ] Add expert pruning for MoE layers (remove entire experts)
- [ ] Add pruning-aware training:
  - [ ] Implement iterative magnitude pruning (IMP)
  - [ ] Add gradual pruning with fine-tuning cycles
  - [ ] Support lottery ticket hypothesis exploration
- [ ] Add pruning metrics:
  - [ ] Track parameter count reduction
  - [ ] Measure FLOPs reduction
  - [ ] Monitor accuracy degradation vs. compression ratio
  - [ ] Generate pruning analysis reports

---

### 3. Ring Attention Removal & Striped Attention Optimization (Priority: HIGH)
**Estimated Effort:** 1-2 weeks

**Rationale:**
- Ring Attention has complex communication overhead unsuitable for edge devices
- Striped Attention already implemented and more efficient for edge deployment
- Ring Attention requires multi-GPU setup, limiting edge applicability
- Maintenance burden of supporting both implementations

#### Action Items:
- [ ] **Remove Ring Attention implementation:**
  - [ ] Delete or deprecate `better_ai/models/ring_attention.py`
  - [ ] Remove Ring Attention imports from `better_ai/models/__init__.py`
  - [ ] Remove Ring Attention config options from `better_ai/config.py`
  - [ ] Update documentation to indicate Striped Attention as primary long-context solution
- [ ] **Optimize Striped Attention for edge deployment:**
  - [ ] Implement INT8 quantization for attention computation
  - [ ] Add flash attention integration for memory efficiency
  - [ ] Optimize for CPU-only inference (important for edge)
  - [ ] Add chunked attention computation for memory-constrained devices
  - [ ] Profile memory usage on edge hardware (Raspberry Pi, Jetson Nano)
- [ ] **Enhance Striped Attention features:**
  - [ ] Add dynamic stripe width based on available memory
  - [ ] Implement adaptive context window sizing
  - [ ] Add KV cache compression for long contexts
  - [ ] Support sliding window attention as fallback
- [ ] **Update training pipeline:**
  - [ ] Remove Ring Attention references from trainer configurations
  - [ ] Update distributed training to use only Striped Attention
  - [ ] Verify training stability without Ring Attention
- [ ] **Add benchmarks:**
  - [ ] Compare Striped vs. Ring Attention on edge devices
  - [ ] Measure throughput, latency, and memory usage
  - [ ] Document performance improvements

---

## 📋 INCOMPLETE FEATURES (High Priority)

### Phase 4: Training Pipeline Integration (40% Complete)
**Estimated Effort:** 2-3 weeks

#### 4.1 Multi-Stage Training Strategy
**Status:** Partial implementation, needs completion

- [x] Pretraining with Stack v2 - Basic implementation
- [x] SFT with Magicoder + Code-Feedback - Working
- [ ] **RLHF Stage 1 with CodeUltraFeedback + GRPO:**
  - [ ] Integrate preference data from CodeUltraFeedback properly
  - [ ] Verify GRPO advantage estimation on code tasks
  - [ ] Add multi-turn dialogue support for coding assistance
  - [ ] Test on actual coding benchmarks (HumanEval, MBPP)
- [ ] **RLHF Stage 2 with multi-attribute regression:**
  - [ ] Implement quantile regression heads (currently minimal)
  - [ ] Add multiple quality dimensions (correctness, efficiency, style, readability)
  - [ ] Train separate reward models for each attribute
  - [ ] Combine attributes into unified reward signal
- [ ] **Iterative refinement with recursive scratchpad:**
  - [ ] Integrate MCTS CoT data into training loop
  - [ ] Add self-correction examples to training data
  - [ ] Implement reasoning trace replay for improved samples

#### 4.2 Curriculum Learning
**Status:** Framework exists, needs tuning

- [ ] **Fine-tune curriculum progression:**
  - [ ] Current cosine schedule may be too fast for edge models
  - [ ] Test different grokking_fast_ratio values (currently 0.4)
  - [ ] Validate plateau_steps effectiveness
  - [ ] Add early stopping based on curriculum convergence
- [ ] **Improve difficulty estimation:**
  - [ ] Current length-based proxy is crude
  - [ ] Add AST complexity metrics (nesting depth, cyclomatic complexity)
  - [ ] Implement perplexity-based difficulty scoring
  - [ ] Use model confidence as difficulty indicator
- [ ] **Enhance domain mixing:**
  - [ ] Current weights may not be optimal for edge deployment
  - [ ] Add domain-specific performance tracking
  - [ ] Implement dynamic weight adjustment based on evaluation metrics
  - [ ] Test mixing ratios on actual edge hardware

#### 4.3 Distributed Training Optimization
**Status:** Basic implementation, needs edge-specific optimization

- [ ] **Optimize for single-device training (edge focus):**
  - [ ] Gradient accumulation with very small batches (batch_size=1)
  - [ ] Memory-efficient optimizer states (8-bit Adam)
  - [ ] Gradient checkpointing to reduce activation memory
  - [ ] Test on consumer GPUs (RTX 3060, M1 Max, etc.)
- [ ] **Improve multi-device training (optional):**
  - [ ] Remove Ring Attention distributed training code
  - [ ] Keep only data-parallel and pipeline-parallel options
  - [ ] Add ZeRO-2/ZeRO-3 offloading for limited VRAM

---

### Phase 7.8: Inference-Time Cosine Curriculum (Priority: MEDIUM)
**Status:** ⏳ Marked as INCOMPLETE in current TODO  
**Estimated Effort:** 1-2 weeks

**Why Critical for Edge:**
- Allows dynamic task difficulty adjustment during inference
- Enables progressive reasoning for complex queries
- Reduces compute for simple queries (saves power on edge)
- Improves accuracy on multi-step problems

#### Action Items:
- [ ] **Extend CosineCurriculumScheduler:**
  - [ ] Add inference mode flag to scheduler
  - [ ] Implement single-query difficulty progression
  - [ ] Support step-wise complexity increase within one generation
- [ ] **Design inference-appropriate metrics:**
  - [ ] Token generation speed as difficulty indicator
  - [ ] Entropy/uncertainty as complexity signal
  - [ ] Question length and structure analysis
- [ ] **Integrate with generation pipeline:**
  - [ ] Add curriculum-aware generation in `better_ai/models/generation.py`
  - [ ] Support dynamic max_tokens based on difficulty
  - [ ] Implement early stopping for easy queries
- [ ] **Combine with MCTS:**
  - [ ] Use curriculum to guide MCTS exploration budget
  - [ ] Easy queries: shallow search, Hard queries: deep search
  - [ ] Add curriculum-weighted UCB exploration
- [ ] **Add evaluation mode:**
  - [ ] Compare curriculum vs. fixed-difficulty inference
  - [ ] Measure accuracy improvement and latency overhead
  - [ ] Test on edge hardware with power monitoring

---

### Phase 7.14-7.15: Data Mixing & Seed Data (Priority: MEDIUM)
**Status:** Not implemented  
**Estimated Effort:** 1 week

#### 7.14 Instruction-Following + Multi-Turn Data Mixing
**Goal:** 75% single-turn, 25% multi-turn dialogue for balanced learning

- [ ] Modify data loading pipeline:
  - [ ] Tag datasets as single-turn vs multi-turn in datasets.yml
  - [ ] Implement sampling with 75/25 ratio
  - [ ] Ensure ratio maintained across epochs
- [ ] Add multi-turn specific processing:
  - [ ] Proper conversation history formatting
  - [ ] Attention mask handling for multi-turn context
  - [ ] Loss masking to avoid learning from previous turns
- [ ] Evaluate mixing effectiveness:
  - [ ] Test single-turn performance (HumanEval)
  - [ ] Test multi-turn performance (MT-Bench style)
  - [ ] Compare vs. single-turn-only and multi-turn-only baselines

#### 7.15 Seed Data with Varying Length Distributions
**Goal:** Small high-quality dataset with diverse response lengths for self-improvement

- [ ] Create seed dataset:
  - [ ] Curate 500-1000 examples with length distribution: short (10-50 tokens), medium (50-200), long (200-500), very long (500-2000)
  - [ ] Ensure high quality (human-written or GPT-4 generated)
  - [ ] Cover diverse domains (code, math, reasoning, general)
- [ ] Implement self-improvement loop:
  - [ ] Generate synthetic data using seed examples as in-context prompts
  - [ ] Filter generated data for quality (use reward model)
  - [ ] Iteratively expand dataset while maintaining length distribution
- [ ] Integrate with training:
  - [ ] Add seed data to SFT stage with high sampling weight
  - [ ] Use seed data for length-aware curriculum initialization
  - [ ] Track length distribution drift during training

---

## 🚀 PHASE 8: INFERENCE OPTIMIZATION & DEPLOYMENT (Priority: CRITICAL)
**Status:** 0% Complete - Completely Missing  
**Estimated Effort:** 3-4 weeks

### Overview:
Phase 8 is **critical for edge deployment and production use**. Without these optimizations, the model cannot run efficiently on edge devices or integrate with popular inference frameworks.

---

### 8.1 Quantization for Edge Devices (Priority: CRITICAL)
**Estimated Effort:** 1 week

**Why Critical:**
- Current FP16/BF16 model is too large for edge devices (1B params = 2GB+ VRAM)
- INT8 quantization = 2-4x smaller, INT4 = 4-8x smaller
- Inference latency reduction: 2-3x speedup on CPU, 1.5-2x on GPU
- Power consumption reduction critical for battery-powered devices

#### Action Items:
- [ ] **Implement INT8 quantization:**
  - [ ] Add post-training quantization (PTQ) using PyTorch quantization API
  - [ ] Support quantization-aware training (QAT) for better accuracy
  - [ ] Quantize all Linear layers, embeddings, and attention
  - [ ] Keep sensitive layers (LayerNorm, softmax) in FP16
- [ ] **Implement INT4 quantization:**
  - [ ] Add GPTQ-style quantization for extreme compression
  - [ ] Support mixed INT4/INT8 (INT4 for FFN, INT8 for attention)
  - [ ] Add k-means codebook quantization for MoE experts
- [ ] **Add dynamic quantization:**
  - [ ] Quantize weights statically, activations dynamically
  - [ ] Implement per-channel quantization for better accuracy
  - [ ] Add outlier-aware quantization (SmoothQuant)
- [ ] **Optimize KV cache quantization:**
  - [ ] Quantize KV cache to INT8 (huge memory savings for long context)
  - [ ] Add per-token quantization scales
  - [ ] Test accuracy impact on 32k+ context lengths
- [ ] **Benchmark quantization:**
  - [ ] Measure accuracy degradation on benchmarks (HumanEval, SWE-bench)
  - [ ] Profile inference speed on edge devices
  - [ ] Compare FP16 vs INT8 vs INT4 memory usage and latency

---

### 8.2 Model Export for Inference Frameworks (Priority: CRITICAL)
**Estimated Effort:** 1-2 weeks

**Why Critical:**
- Users need to run models with popular tools (Ollama, vLLM, llama.cpp)
- Current PyTorch model not optimized for production inference
- GGUF format enables CPU-only inference on consumer hardware
- vLLM provides 10-20x throughput improvement for serving

#### 8.2.1 Export to GGUF (for Ollama, llama.cpp)
**Estimated Effort:** 5-7 days

- [ ] **Implement GGUF converter:**
  - [ ] Add conversion script: `scripts/convert_to_gguf.py`
  - [ ] Map DeepSeek model architecture to GGUF format
  - [ ] Handle MoE expert weights correctly
  - [ ] Export tokenizer to GGUF-compatible format
- [ ] **Support quantized GGUF:**
  - [ ] Export Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 quantization formats
  - [ ] Add K-quant formats (Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K)
  - [ ] Test each quantization level for accuracy
- [ ] **Add model metadata:**
  - [ ] Include model card information in GGUF file
  - [ ] Add chat template for instruction following
  - [ ] Set proper context length and rope settings
- [ ] **Test with Ollama:**
  - [ ] Create Modelfile for Ollama import
  - [ ] Test `ollama run better-ai` locally
  - [ ] Verify generation quality matches PyTorch version
  - [ ] Add to Ollama model library (optional)
- [ ] **Test with llama.cpp:**
  - [ ] Test loading with `llama-cli`
  - [ ] Verify perplexity matches expected values
  - [ ] Test CPU-only inference on various hardware
  - [ ] Add server mode testing (`llama-server`)

#### 8.2.2 Export to vLLM Format
**Estimated Effort:** 5-7 days

- [ ] **Add vLLM compatibility layer:**
  - [ ] Implement `VLLMDeepSeekModel` wrapper in `better_ai/inference/vllm_compat.py`
  - [ ] Map model config to vLLM ModelConfig
  - [ ] Handle MoE routing for vLLM execution
  - [ ] Add KV cache management compatible with vLLM
- [ ] **Support vLLM optimizations:**
  - [ ] Enable PagedAttention for efficient KV cache
  - [ ] Add continuous batching support
  - [ ] Implement speculative decoding (if model supports it)
  - [ ] Support tensor parallelism for multi-GPU deployment
- [ ] **Create vLLM serving config:**
  - [ ] Add example `config.json` for vLLM model loading
  - [ ] Configure chat template and generation parameters
  - [ ] Set optimal batch size and memory limits
- [ ] **Test vLLM deployment:**
  - [ ] Test `vllm serve` with model
  - [ ] Measure throughput (tokens/sec) vs. PyTorch baseline
  - [ ] Test with multiple concurrent requests
  - [ ] Verify generation quality unchanged

#### 8.2.3 Export to TensorRT-LLM (NVIDIA Optimization)
**Estimated Effort:** 3-5 days (Optional)

- [ ] **Add TensorRT-LLM build:**
  - [ ] Create TensorRT-LLM conversion script
  - [ ] Build optimized engine for target GPU (A100, 4090, etc.)
  - [ ] Enable FP16/INT8/FP8 inference
- [ ] **Optimize for NVIDIA hardware:**
  - [ ] Use Flash Attention 2 for attention optimization
  - [ ] Enable tensor parallelism for large contexts
  - [ ] Add in-flight batching for serving
- [ ] **Benchmark TensorRT-LLM:**
  - [ ] Compare vs. PyTorch and vLLM
  - [ ] Measure latency and throughput
  - [ ] Test on edge GPUs (Jetson Orin, etc.)

#### 8.2.4 Export to CoreML (Apple Silicon)
**Estimated Effort:** 3-5 days (Optional)

- [ ] **Add CoreML conversion:**
  - [ ] Use coremltools to convert model
  - [ ] Optimize for Apple Neural Engine
  - [ ] Test on M1/M2/M3 chips
- [ ] **Optimize for Metal GPU:**
  - [ ] Enable Metal Performance Shaders
  - [ ] Test on iPhone/iPad (optional)

---

### 8.3 KV Cache Optimization (Priority: HIGH)
**Estimated Effort:** 5-7 days

**Problems:**
- Current KV cache grows linearly with context length (OOM on long contexts)
- No cache compression or eviction strategy
- Memory usage prevents long-context inference on edge devices

#### Action Items:
- [ ] **Implement KV cache compression:**
  - [ ] Add INT8 quantization for KV cache (implemented in 8.1)
  - [ ] Implement H2O (Heavy-Hitter Oracle) for cache eviction
  - [ ] Add StreamingLLM for infinite context (keep recent + initial tokens)
  - [ ] Test on 32k, 64k, 128k context lengths
- [ ] **Add sliding window attention:**
  - [ ] Implement fixed-size attention window (e.g., 4096 tokens)
  - [ ] Keep only recent tokens in cache
  - [ ] Add global tokens (summary of old context)
- [ ] **Implement cache sharing for batching:**
  - [ ] Share KV cache across batch items with common prefix
  - [ ] Add copy-on-write for cache management
  - [ ] Test with multi-turn conversations
- [ ] **Add cache persistence:**
  - [ ] Save/load KV cache to/from disk for long conversations
  - [ ] Implement cache warm-up for frequent prompts
  - [ ] Add cache TTL and LRU eviction

---

### 8.4 Memory Management & Optimization (Priority: HIGH)
**Estimated Effort:** 4-5 days

**Problems:**
- No memory profiling or monitoring
- Activation memory grows unbounded during generation
- Peak memory usage not optimized

#### Action Items:
- [ ] **Add memory profiling:**
  - [ ] Integrate PyTorch memory profiler
  - [ ] Track peak memory usage per operation
  - [ ] Add memory usage logging during inference
  - [ ] Generate memory usage reports
- [ ] **Implement gradient checkpointing for inference:**
  - [ ] Use activation checkpointing to reduce memory
  - [ ] Trade compute for memory (useful on edge)
  - [ ] Test memory reduction vs. latency increase
- [ ] **Add memory-efficient generation:**
  - [ ] Implement in-place operations where possible
  - [ ] Reduce intermediate tensor allocations
  - [ ] Use torch.cuda.empty_cache() strategically
- [ ] **Optimize for low-memory devices:**
  - [ ] Add "memory_limit" parameter to auto-adjust batch size
  - [ ] Implement out-of-core computation for very large models
  - [ ] Test on devices with 4GB, 8GB, 16GB VRAM

---

### 8.5 API Compatibility Layer (Priority: HIGH)
**Estimated Effort:** 4-5 days

**Why Critical:**
- Users expect OpenAI-compatible API for easy integration
- Enable drop-in replacement for ChatGPT API
- Support streaming responses for better UX

#### Action Items:
- [ ] **Implement OpenAI-compatible API:**
  - [ ] Add FastAPI server in `better_ai/inference/api_server.py`
  - [ ] Implement `/v1/chat/completions` endpoint
  - [ ] Implement `/v1/completions` endpoint
  - [ ] Support streaming responses (SSE)
- [ ] **Add API features:**
  - [ ] Support temperature, top_p, max_tokens parameters
  - [ ] Implement stop sequences
  - [ ] Add function calling / tool use support
  - [ ] Support multiple messages in conversation
- [ ] **Add authentication & rate limiting:**
  - [ ] API key authentication
  - [ ] Rate limiting per user/API key
  - [ ] Usage tracking and logging
- [ ] **Create Python client library:**
  - [ ] OpenAI-compatible client wrapper
  - [ ] Async support for streaming
  - [ ] Automatic retries and error handling
- [ ] **Add deployment configs:**
  - [ ] Docker container for API server
  - [ ] Kubernetes deployment YAML
  - [ ] Environment variable configuration
  - [ ] Health check endpoints

---

### 8.6 RAG (Retrieval-Augmented Generation) Support (Priority: MEDIUM)
**Estimated Effort:** 1-2 weeks

**Why Useful:**
- Enhance code generation with relevant documentation
- Support repository-aware code completion
- Enable scientific paper retrieval for research tasks

#### Action Items:
- [ ] **Implement vector database integration:**
  - [ ] Add support for FAISS, Qdrant, or ChromaDB
  - [ ] Create embedding generation pipeline
  - [ ] Implement efficient similarity search
- [ ] **Add document processing:**
  - [ ] Support markdown, code, and PDF ingestion
  - [ ] Implement chunking strategies (fixed-size, semantic)
  - [ ] Add metadata extraction (language, file path, etc.)
- [ ] **Create RAG pipeline:**
  - [ ] Query → Retrieve → Rerank → Generate
  - [ ] Add relevance scoring for retrieved documents
  - [ ] Implement context compression for long documents
  - [ ] Support multi-hop retrieval for complex queries
- [ ] **Add code-specific RAG:**
  - [ ] Index entire codebases (AST-aware chunking)
  - [ ] Retrieve relevant functions/classes for coding tasks
  - [ ] Add import statement resolution
  - [ ] Support cross-file code understanding
- [ ] **Optimize for edge deployment:**
  - [ ] Use quantized embeddings for smaller index
  - [ ] Implement on-device vector search (no external DB)
  - [ ] Add local cache for frequently retrieved documents

---

### 8.7 Inference Benchmarking & Profiling (Priority: MEDIUM)
**Estimated Effort:** 3-4 days

#### Action Items:
- [ ] **Add comprehensive benchmarks:**
  - [ ] Create `better_ai/inference/benchmark.py`
  - [ ] Measure tokens/second for various batch sizes
  - [ ] Test prefill vs. decode latency separately
  - [ ] Benchmark on multiple context lengths (512, 2k, 8k, 32k)
- [ ] **Add hardware-specific benchmarks:**
  - [ ] Test on edge devices: Raspberry Pi 5, Jetson Orin Nano, Intel NUC
  - [ ] Test on consumer GPUs: RTX 3060, 4070, 4090
  - [ ] Test on Apple Silicon: M1, M2, M3 (various RAM sizes)
  - [ ] Test on cloud GPUs: A100, H100, L4, T4
- [ ] **Add power profiling:**
  - [ ] Measure power consumption on battery-powered devices
  - [ ] Track energy per token generated
  - [ ] Compare quantized vs. full-precision power usage
- [ ] **Generate performance reports:**
  - [ ] Create markdown reports with tables and charts
  - [ ] Compare vs. similar-sized models (1B-3B params)
  - [ ] Add recommendations for optimal deployment config

---

## 🐛 CODE QUALITY ISSUES (Medium Priority)

### Issue 1: CodeDataset Implementation Problems
**File:** `better_ai/data/datasets/code_dataset.py` (TO BE DELETED)
**Priority:** MEDIUM (will be resolved by deletion)

**Problems:**
- Brittle language filtering: Uses string matching on `'python' in language.lower()` (fails for "python3", "cpython")
- Inefficient iteration: Iterates entire dataset to filter, doesn't use dataset.filter()
- Naive code cleaning: Simple line-based cleaning, no AST validation
- Hardcoded limits: Max 500 lines, 10K chars (should be configurable)
- Poor error handling: Falls back to synthetic data on any error
- Synthetic data is toy-level quality, not suitable for training

**Resolution:** Delete file as part of dataset migration (Critical Issue #1)

---

### Issue 2: Striped Attention Optimization Gaps
**File:** `better_ai/models/attention.py`  
**Priority:** HIGH (now primary long-context solution)

**Problems:**
- No dynamic stripe width adjustment based on available memory
- Missing flash attention integration
- No profiling instrumentation
- Lacks CPU-optimized path for edge inference

**Action Items (moved to Critical Issue #3):**
- See "Ring Attention Removal & Striped Attention Optimization" section above

---

### Issue 3: Inconsistent Naming & Style
**Priority:** LOW  
**Estimated Effort:** 2-3 days

**Problems:**
- Mixed snake_case and camelCase in config files
- Inconsistent docstring styles (Google vs. NumPy)
- Some functions missing type hints
- Variable naming: `curr` vs `current`, `idx` vs `index`

**Action Items:**
- [ ] Run black formatter on entire codebase
- [ ] Add flake8 and mypy to CI pipeline
- [ ] Standardize docstrings to Google style
- [ ] Add type hints to all public functions
- [ ] Create style guide in CONTRIBUTING.md

---

### Issue 4: Error Handling Improvements
**Priority:** MEDIUM  
**Estimated Effort:** 3-4 days

**Problems:**
- Many functions use bare `except:` (catches KeyboardInterrupt, SystemExit)
- Error messages lack context (which file, which step, what values)
- No structured logging with log levels
- Missing validation of user inputs

**Action Items:**
- [ ] Replace all bare `except:` with specific exceptions
- [ ] Add context to error messages (include variable values)
- [ ] Set up structured logging with Python logging module
- [ ] Add input validation with clear error messages
- [ ] Create custom exception classes for domain-specific errors

---

### Issue 5: Testing Coverage
**Priority:** MEDIUM  
**Estimated Effort:** 1-2 weeks

**Problems:**
- Minimal unit test coverage (tests/ directory exists but sparse)
- No integration tests for training pipeline
- No regression tests for model outputs
- Missing tests for edge cases and error conditions

**Action Items:**
- [ ] Add unit tests for all data loading modules
- [ ] Add unit tests for model components (attention, MoE, etc.)
- [ ] Create integration tests for end-to-end training
- [ ] Add regression tests with frozen model checkpoints
- [ ] Set up pytest with coverage reporting
- [ ] Aim for >80% coverage on core modules

---

## 🔧 EDGE DEPLOYMENT CRITICAL ISSUES

### Issue 1: Model Size Too Large for Edge Devices
**Priority:** CRITICAL (addressed in Phase 8.1)

**Problems:**
- 1B parameter model in FP16 = ~2GB VRAM (excludes KV cache and activations)
- Real inference needs 4-6GB for comfortable operation
- Most edge devices have 4-8GB total memory (shared with OS)
- Raspberry Pi 5: 4-8GB RAM, Jetson Orin Nano: 8GB, iPhone 15: 6-8GB

**Action Items:**
- See Phase 8.1 for quantization implementation
- Target: <1GB model size with INT4 quantization
- Enable inference on 4GB RAM devices

---

### Issue 2: Inference Latency Optimization
**Priority:** HIGH  
**Estimated Effort:** 1 week

**Problems:**
- No operator fusion or graph optimization
- Missing ONNX export for optimized runtimes
- No benchmarking on target edge hardware

**Action Items:**
- [ ] Add torch.jit.script compilation for inference
- [ ] Implement operator fusion (matmul + bias + gelu → single op)
- [ ] Add ONNX export with optimization passes
- [ ] Test with ONNX Runtime on edge devices
- [ ] Profile and optimize bottleneck operations
- [ ] Target: <100ms latency per token on Raspberry Pi 5

---

### Issue 3: Power Consumption Monitoring
**Priority:** MEDIUM  
**Estimated Effort:** 3-4 days

**Problems:**
- No power profiling for battery-powered deployment
- Unknown battery life for mobile deployment
- Missing energy-efficient inference modes

**Action Items:**
- [ ] Add power monitoring using NVIDIA SMI / jetson-stats / powerstat
- [ ] Measure energy per token for different configurations
- [ ] Implement "eco mode" with reduced quality for longer battery life
- [ ] Test battery life on actual mobile devices
- [ ] Add power usage to benchmarking reports

---

### Issue 4: On-Device Fine-Tuning Support
**Priority:** LOW (Future Work)  
**Estimated Effort:** 2-3 weeks

**Goal:** Enable personalization and adaptation on edge devices

**Action Items:**
- [ ] Implement LoRA for parameter-efficient fine-tuning
- [ ] Add QLoRA for INT4-quantized model fine-tuning
- [ ] Create on-device training pipeline with tiny batches
- [ ] Test fine-tuning on Jetson Orin Nano (8GB)
- [ ] Add federated learning support (optional)

---

## 📊 MONITORING & OBSERVABILITY (Medium Priority)
**Estimated Effort:** 1 week

### Action Items:
- [ ] **Add training metrics dashboard:**
  - [ ] Integrate TensorBoard or Weights & Biases
  - [ ] Track loss, perplexity, learning rate, gradient norms
  - [ ] Add custom metrics (diversity, reasoning quality, etc.)
  - [ ] Log expert utilization for MoE analysis
- [ ] **Add inference monitoring:**
  - [ ] Track latency (p50, p95, p99)
  - [ ] Monitor tokens/second throughput
  - [ ] Log memory usage and cache hit rates
  - [ ] Add error rate tracking
- [ ] **Create alerting system:**
  - [ ] Alert on training divergence (loss NaN, gradient explosion)
  - [ ] Alert on inference failures
  - [ ] Alert on memory usage exceeding limits

---

## 🔒 SECURITY & SAFETY (Low Priority, Post-MVP)
**Estimated Effort:** 1-2 weeks

### Action Items:
- [ ] **Add input sanitization:**
  - [ ] Validate and sanitize user prompts
  - [ ] Prevent prompt injection attacks
  - [ ] Add content filtering (optional, configurable)
- [ ] **Implement output safety:**
  - [ ] Add safety classifier for generated code
  - [ ] Detect potentially harmful outputs (rm -rf, etc.)
  - [ ] Add warning flags for risky operations
- [ ] **Add model security:**
  - [ ] Model watermarking for attribution
  - [ ] Adversarial robustness testing
  - [ ] Backdoor detection in training data

---

## 📈 PERFORMANCE TARGETS

### Training Performance:
- [ ] Pretraining: >5000 tokens/sec on single A100
- [ ] SFT: >3000 tokens/sec on single A100
- [ ] RLHF: >1000 tokens/sec (accounting for multiple rollouts)
- [ ] Memory usage: <40GB VRAM for full training
- [ ] Convergence: Match or exceed baseline within 20% training time

### Inference Performance (Edge Devices):
- [ ] **Raspberry Pi 5 (INT4):**
  - Latency: <150ms/token
  - Throughput: >6 tokens/sec
  - Memory: <2GB total
- [ ] **Jetson Orin Nano (INT8):**
  - Latency: <50ms/token
  - Throughput: >20 tokens/sec
  - Memory: <4GB total
- [ ] **Apple M2 (INT8):**
  - Latency: <30ms/token
  - Throughput: >30 tokens/sec
  - Memory: <3GB total
- [ ] **RTX 4090 (FP16):**
  - Latency: <10ms/token
  - Throughput: >100 tokens/sec
  - Memory: <12GB total

### Model Quality Targets:
- [ ] HumanEval pass@1: >45% (target: 50%)
- [ ] MBPP pass@1: >50% (target: 55%)
- [ ] SWE-bench Lite: >15% (target: 20%)
- [ ] GSM8K: >70% (target: 75%)
- [ ] MATH: >30% (target: 35%)
- [ ] MT-Bench: >7.0 (target: 7.5)

---

## 🗓️ EFFORT SUMMARY

### Critical Priority (Must Complete for MVP):
- Dataset System Migration: 3-5 days
- Stub Implementations: 2-3 weeks
- Ring Attention Removal & Striped Optimization: 1-2 weeks
- Phase 8 (Inference & Export): 3-4 weeks
- **Total Critical: 7-10 weeks**

### High Priority (Important for Quality):
- Training Pipeline Completion: 2-3 weeks
- Memory & KV Cache Optimization: 1-2 weeks
- Inference Latency Optimization: 1 week
- **Total High: 4-6 weeks**

### Medium Priority (Quality of Life):
- Inference-Time Curriculum: 1-2 weeks
- Data Mixing & Seed Data: 1 week
- Code Quality & Testing: 2-3 weeks
- Monitoring & Observability: 1 week
- **Total Medium: 5-7 weeks**

### Low Priority (Post-MVP):
- Naming & Style Consistency: 2-3 days
- On-Device Fine-Tuning: 2-3 weeks
- Security & Safety: 1-2 weeks
- **Total Low: 3-5 weeks**

### **Grand Total Estimated Effort: 19-28 weeks (~5-7 months)**

---

## 📝 NOTES

### For Agentic Coding:
- Each task includes clear success criteria and implementation details
- File paths provided for easy navigation
- Problems clearly described with context
- Estimated effort helps prioritization
- Dependencies noted where relevant

### For Edge Deployment:
- All Phase 8 tasks are critical for edge viability
- Quantization (8.1) and Export (8.2) are highest priority
- Memory optimization (8.3, 8.4) required for 4GB devices
- Benchmarking (8.7) validates all optimizations

### For Scientific Tasks:
- RAG support (8.6) enables research paper retrieval
- Long context (via Striped Attention) crucial for scientific documents
- Reasoning features (MCTS, ToT) already implemented
- Verification systems (8.6) support mathematical proofs

---

## 🚀 GETTING STARTED

### Recommended Order:
1. **Week 1-2:** Dataset migration + Ring Attention removal (cleanup foundation)
2. **Week 3-5:** Stub implementations (fault localization, diversity, pruning)
3. **Week 6-8:** Phase 8.1 Quantization (enable edge deployment)
4. **Week 9-10:** Phase 8.2 Model Export (GGUF, vLLM)
5. **Week 11-12:** Phase 8.3-8.4 Memory & KV Cache optimization
6. **Week 13-15:** Complete Phase 4 Training Pipeline
7. **Week 16-19:** Polish (testing, monitoring, benchmarking)

### Quick Wins (Do First):
- [ ] Add input validation with helpful error messages
- [ ] Set up GitHub Issues for each TODO item
- [ ] Add basic logging to all training scripts
- [ ] Create Docker container for reproducible setup
- [ ] Write usage examples for each major feature

---

**Last Updated:** February 14, 2026  
**Version:** 1.0.0  
**Maintainer:** Better AI Team