# Better AI - TODO (Incomplete Tasks Only)

**Last Updated:** February 16, 2026  
**Status:** Active Development - MoE Optimization Focus

> See [CHANGELOG.md](CHANGELOG.md) for comprehensive list of completed features.

---

## 🚨 CURRENT SPRINT: MoE Memory Optimizations

### In Progress
- [ ] **Optimization 1**: Router logit chunked computation (70-80% router memory reduction)
- [ ] **Optimization 2**: Expert output buffer pooling (50% forward pass memory reduction)
- [ ] **Optimization 3**: Fused softmax-topk operations (40-60% routing memory reduction)
- [ ] **Optimization 4**: Batched expert processing with grouped GEMM (30-40% memory + 2-4x speedup)
- [ ] **Optimization 5**: Dynamic expert pruning during inference (40-60% inference memory reduction)

---

## 🔧 INCOMPLETE FEATURES

### Phase 1: Striped Attention Refinements
- [ ] Implement real distributed ring communication (currently mocked/gathered)
- [ ] Support dynamic stripe width based on available VRAM
- [ ] Add flash attention integration to striped kernels

### Phase 1.1: Advanced Core Refinements
- [ ] **Flash Attention 2/3 Explicit Integration**:
  - [ ] Add explicit support for `flash-attn` library if installed
  - [ ] Implement custom CUDA kernels for LSE (Log-Sum-Exp) output
  - [ ] Integrate xformers' memory-efficient attention as alternative backend
- [ ] **GQA Memory-Optimized Sharing**:
  - [ ] Refactor `StripedAttention` and `FlashMultiHeadAttention` to avoid KV repetition
  - [ ] Implement grouped query reshaping to leverage broadcasting
- [ ] **RoPE Context Scaling (YaRN/NTK)**:
  - [ ] Implement YaRN (Yet another RoPE extension) in `better_ai/models/rope.py`
  - [ ] Add NTK-aware scaling with dynamic base adjustment
- [ ] **Expert Choice Routing**:
  - [ ] Complete `ExpertChoiceRouter` implementation in `better_ai/models/moe_optimized.py`
  - [ ] Implement specialized expert-centric forward pass
- [ ] **V-STaR (Verifier-based STaR)**:
  - [ ] Integrate trained verifier model into `STaRModule`
  - [ ] Implement real-world verification loops (Python execution, math solvers)
- [ ] **Advanced MCTS (PUCT/UCB1)**:
  - [ ] Implement PUCT (Predictor + Upper Confidence Bound) for node selection
  - [ ] Add transposition tables to `mcts_cot.py` for state caching

### Phase 4: Training Enhancements
- [ ] **Recursive Scratchpad Iteration**: Refine MCTS CoT integration
- [ ] **Curriculum Fine-tuning**:
  - [ ] Fine-tune curriculum progression rates using real datasets
  - [ ] Test different `grokking_fast_ratio` values (currently 0.4)
  - [ ] Validate plateau_steps effectiveness on real-world tasks
  - [ ] Add scripts for automated curriculum tuning

### Phase 4.1: RLHF Refinements
- [ ] **Rejection Sampling**:
  - [ ] Modify `grpo.py` to support N-solution rollouts per prompt
  - [ ] Implement reward-based filtering (top-K selection)
- [ ] **Data Contamination Detection**:
  - [ ] Implement n-gram (8-13) overlap detection in `curation.py`
  - [ ] Add embedding-based similarity check for benchmark leakage
- [ ] **RLVR Context Training**:
  - [ ] Implement Stage 5: RLVR training with `[CONTEXT]` tags
  - [ ] Train model to prioritize instructions while leveraging context
- [ ] **Standardized Evaluation Harness**:
  - [ ] Integrate `lm-evaluation-harness` for MMLU, GSM8K, etc.

### Phase 5: Benchmarking
- [ ] **Performance Benchmarking**: Comparative analysis vs baseline
- [ ] **Optimization Profiling**: Memory-efficient variants

### Phase 6: Multi-Modal
- [ ] **Visual Alignment Refinement**:
  - [ ] Replace MLP with cross-attention in `VisualAlignmentLayer`
  - [ ] Optimize visual token projection for edge inference
- [ ] **Full Multi-Modal Training**: Align vision encoder with LLM backbone

### Phase 7: Safety
- [ ] **Adversarial Red Teaming**:
  - [ ] Jailbreak protection
  - [ ] Safety guardrails for harmful code generation

### Phase 8: Inference Optimization
- [ ] **Quantization**:
  - [ ] Complete `apply_int8_quantization` with `torch.ao.quantization`
  - [ ] Support quantization-aware training (QAT)
  - [ ] Implement INT4 quantization (GPTQ-style)
  - [ ] Mixed INT4/INT8 (INT4 for FFN, INT8 for attention)
- [ ] **Model Export**:
  - [ ] Map DeepSeek architecture to GGUF format
  - [ ] Handle MoE expert weights correctly in GGUF
  - [ ] Export tokenizer to GGUF-compatible format
  - [ ] Enable PagedAttention for vLLM
  - [ ] Support continuous batching
  - [ ] Create TensorRT-LLM conversion script
  - [ ] Use coremltools for Apple Neural Engine
- [ ] **KV Cache**:
  - [ ] Add sliding window attention with global tokens
  - [ ] Implement cache sharing for batching (common prefix)
- [ ] **Memory Management**:
  - [ ] Implement gradient checkpointing for inference
  - [ ] Add `memory_limit` parameter for auto-adjust batch size
- [ ] **API Refinement**:
  - [ ] Replace mock tool-calling with real orchestration
  - [ ] Integrate real DeepSeek model loading (replace MockModel)
  - [ ] Support temperature, top_p, max_tokens parameters
  - [ ] Add function calling / tool use support
  - [ ] Authentication & rate limiting
- [ ] **RAG Enhancements**:
  - [ ] Integrate Sentence-Transformers for embeddings
  - [ ] Implement semantic chunking
  - [ ] Add vector database integration (FAISS, Qdrant)
  - [ ] Create embedding generation pipeline
  - [ ] Index entire codebases with AST-aware chunking
- [ ] **Benchmarking**:
  - [ ] Test on Raspberry Pi 5, Jetson Orin Nano, Apple Silicon
  - [ ] Measure energy per token generated

---

## 🐛 CODE QUALITY

- [ ] **Striped Attention Optimization Gaps**:
  - [ ] No dynamic stripe width adjustment
  - [ ] Missing flash attention integration
- [ ] **Inconsistent Naming & Style**:
  - [ ] Run black formatter on entire codebase
  - [ ] Add flake8 and mypy to CI pipeline
- [ ] **Error Handling**:
  - [ ] Replace all bare `except:` with specific exceptions
  - [ ] Add context to error messages
- [ ] **Production-grade Fault Localization**:
  - [ ] Implement specialized reasoning heads for fault prediction
  - [ ] Use cross-modal attention for error trace alignment
  - [ ] Integrate external static analysis tools

---

## 🚀 EDGE DEPLOYMENT

- [ ] **Model Size**: Target <1GB with INT4 quantization
- [ ] **Inference Latency**:
  - [ ] Add `torch.jit.script` compilation
  - [ ] Implement operator fusion
- [ ] **Power Consumption**: Measure energy per token

---

## 📊 MONITORING

- [ ] **Training Metrics Dashboard**:
  - [ ] Integrate TensorBoard or Weights & Biases
- [ ] **Inference Monitoring**:
  - [ ] Track latency (p50, p95, p99)
  - [ ] Log cache hit rates

---

## 🔒 SECURITY

- [ ] **Output Safety**:
  - [ ] Add safety classifier for generated code
  - [ ] Detect potentially harmful outputs (rm -rf, etc.)

---

## 📈 PERFORMANCE TARGETS

- [ ] **Latency**: <100ms per token on Raspberry Pi 5
- [ ] **Accuracy**: >80% on HumanEval (Pass@1)
- [ ] **Convergence**: 20% faster than baseline MoE

---

**Maintainer:** Better AI Team  
**Version:** 2.1.0
