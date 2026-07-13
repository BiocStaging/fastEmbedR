# Third-party source licenses

fastEmbedR is distributed under the MIT license. The native nearest-neighbour
implementation contains permissively licensed derivative work with the
following provenance:

- **FAISS 1.14.3**, commit
  `0ca9df4792b173d573044ee14ca0704780176e82`. Copyright Meta Platforms,
  Inc. and affiliates. MIT license: `FAISS-LICENSE`.
- **MLXPorts/Faiss-mlx**, commit
  `d092af559375144fc719cd88a10e414f92c625fa`. Copyright 2024 Sydney Bach,
  The Solace Project. Apache License 2.0: `FAISS-MLX-LICENSE`.
- **RAPIDS cuVS**, linked optionally through its stable C API. Copyright
  NVIDIA Corporation. Apache License 2.0: `CUVS-LICENSE`. cuVS source and
  binaries are not bundled in fastEmbedR.
- **DLPack**, stable C tensor ABI declarations used by the cuVS C interface.
  Copyright DMLC contributors. Apache License 2.0: `DLPACK-LICENSE`.

The package source files identify the portions derived from or informed by
these projects. CUDA nearest-neighbour integration remains owned by the
optional faissR package and is not copied into fastEmbedR.
