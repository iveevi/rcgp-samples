# rcgp-samples

Samples for [rcgp](https://github.com/iveevi/rcgp), a C++26 library that puts
shader programming and resource declarations in the host language, so that
resource misuse is caught at compile time.

## Building

Requires a C++26 compiler (GCC 15+ / Clang 19+), CMake 4.0+, Vulkan, glm and
glfw.

```sh
git clone --recursive https://github.com/iveevi/rcgp-samples
cd rcgp-samples
just build            # or: cmake -B build && cmake --build build
```

Build one sample with `just sample 04_textures`. Binaries land in `build/`.

## Samples

Each sample is a single self-contained file in `samples/`. Ones taking a
`<scene>` argument accept any `.gltf` or `.glb` file.

| | | |
|---|---|---|
| `01_triangle` | | the smallest program: one vertex stream, no descriptors |
| `02_cube` | | indexed geometry, depth, a push constant and a uniform buffer |
| `03_mesh` | `<scene>` | the same, over geometry loaded from a glTF file |
| `04_textures` | | base-color and ARM maps sampled through `Sampler2D` |

More samples are on the way.
