# LinkedIn Post Draft - F16 GPU OpenCL Follow-up

---

Quick follow-up on my HAT Llama 3.2 inference experiment - I got native FP16 (half-precision) weights running on GPU through OpenCL!

The journey to get there was... interesting.

**The starting point**: I had all 6 kernels (GEMV, RMSNorm, RoPE, SiLU, Softmax, Attention) working through HAT @Reflect dispatch on CPU. The natural next step was GPU. How hard could it be?

**Bug #1 - Kernel name collision**: When my HAT kernel methods delegated to shared plain-Java implementations (`GEMVHAT.gemvKernel()` calling `GEMV.gemvKernel()`), Babylon's codegen emitted the same function name as both `HAT_FUNC` and `HAT_KERNEL` in OpenCL C - causing a "redefinition" compile error.

Fix: inline kernel bodies directly in the HAT class. The official nbody example does this too.

**Bug #2 - F16 codegen**: This was the fun one. `F16.f16ToFloat(matrix.array(i))` generated broken OpenCL:

```c
// Generated (broken):
(float)&matrix->array[i]->value
// Should be:
(float)matrix->array[i].value
```

Two syntax errors: `->` instead of `.` (struct, not pointer) and a spurious `&`. My F32 kernels worked fine on GPU (~10x speedup over CPU), but F16 was completely blocked.

I dug into Babylon's codegen pipeline (`HATFP16Phase` -> `OpenCLHATKernelBuilder`) and found that `hatF16ToFloatConvOp` checks `isLocal()` to decide between `->value` (pointer) and `.value` (local struct). When `f16ToFloat` receives an array element directly, the codegen misclassifies it as a pointer.

**The one-line fix**:

```java
// Before (broken on OpenCL):
sum += F16.f16ToFloat(matrix.array(i)) * vector.array(c);

// After (works!):
F16 weight = matrix.array(i);
sum += F16.f16ToFloat(weight) * vector.array(c);
```

Extracting to a local variable makes `isLocal()` return true -> correct `.value` access.

**Result**: Native F16 weights on GPU OpenCL, no conversion to F32 needed.

| Backend          | Tokens/sec | vs CPU F16 |
|-----------------|-----------|------------|
| CPU Java: F16    | 0.02 tok/s | 1x         |
| CPU Java: F32    | 0.04 tok/s | 2x         |
| GPU OpenCL: F16  | 0.28 tok/s | 14x        |
| GPU OpenCL: F32  | 0.41 tok/s | 20x        |

14x speedup with native half-precision weights on a MacBook. Not bad for a "slow proof of concept" 😄

The F16-to-F32 gap on GPU (0.28 vs 0.41) is just the half-precision arithmetic overhead - expected and acceptable given the 50% memory savings.

Full technical writeup with codegen analysis in the repo:
https://github.com/ArturSkowronski/llama3-java-hat

Next steps: integrate @Juan Fumero's Flash Attention (just merged with FP16 support!), try `MINIMIZE_COPIES` flag that @Ana-Maria Mihalceanu suggested, and explore interface mapping for the GGUF reader.

Thanks to the Babylon team (@Ana-Maria Mihalceanu, @Juan Fumero, @Lize Raes) for the encouragement and pointers! The fact that you can debug OpenCL codegen issues by reading actual Java source code in the HAT pipeline is a massive DX win compared to traditional GPU debugging. I felt like a detective, not a victim. 🕵️

#Java #ProjectBabylon #HAT #OpenCL #GPU #FP16 #LLM #MachineLearning

---

## Notes for editing:
- Consider shortening - LinkedIn has ~3000 char limit for non-expanded view, first ~210 chars visible before "see more"
- The code blocks may not render well on LinkedIn mobile - could replace with screenshots
- Tag the actual LinkedIn profiles of Ana, Juan, Lize
- Could add the benchmark table as an image for better formatting
- Opening line should hook - the current one works as a direct follow-up to the thread
