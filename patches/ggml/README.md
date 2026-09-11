# GGML patches

The `ggml/` submodule always points to a commit available from the official
[`ggml-org/ggml`](https://github.com/ggml-org/ggml) repository. Project-specific
changes are kept here as ordered patches so a normal recursive clone never
depends on a private GGML fork or an unreachable commit.

Current base: `8c63e70982c95ceb862e3a1073a2c1beef75d60a` (`v0.20.2`).

- `0001` retains a scalar Vulkan F32 pipeline when cooperative matrices are
  available, allowing `GGML_PREC_F32` graphs to avoid an implicit F16 path.
- `0002` handles F16 weights with F32 inputs on coopmat2 without selecting the
  unavailable F16-by-F32 pipeline.
- `0003` adds a Metal kernel for `GGML_OP_DIAG_MASK_INF` (the causal attention
  mask used by the TokenRig Qwen graph), which upstream's Metal backend no longer
  ships. Without it `--device auto` aborts on Apple GPUs with `unsupported op`.

CMake copies the pristine submodule into the active build directory and applies
these patches there during configuration. It never modifies `ggml/`. Patch
application is checked before compilation and configuration fails if the base
revision or patch context differs.

To refresh the patches after selecting a newer upstream base, apply the old
patches to a temporary GGML branch, rebase and resolve them, then export with:

```sh
git -C ggml format-patch --output-directory ../patches/ggml NEW_BASE..HEAD
```

Return `ggml/` to `NEW_BASE`, update `SKINTOKENS_GGML_BASE_REVISION` in the
top-level `CMakeLists.txt`, and run clean CPU, Vulkan, and sanitizer builds.
