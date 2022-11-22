#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/neural_radiance_field.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/reduce_sum.h>

NGP_NAMESPACE_BEGIN

__global__ void nerf_grid_to_bitfield(
    const uint32_t n_elements,
    const uint32_t n_nonzero_elements,
    const float* __restrict__ grid,
    uint8_t* __restrict__ grid_bitfield,
    const float* __restrict__ mean_density_ptr,
    const float min_optical_thickness
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;
    if (i >= n_nonzero_elements) {
        grid_bitfield[i] = 0;
        return;
    }

    uint8_t bits = 0;

    float thresh = std::min(min_optical_thickness, *mean_density_ptr);

    NGP_PRAGMA_UNROLL
    for (uint8_t j = 0; j < 8; ++j) {
        bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
    }

    grid_bitfield[i] = bits;
}

__global__ void nerf_bitfield_max_pool(
    const uint32_t n_elements,
    const uint32_t grid_size,
    const uint8_t* __restrict__ prev_level,
    uint8_t* __restrict__ next_level
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    uint8_t bits = 0;

    NGP_PRAGMA_UNROLL
    for (uint8_t j = 0; j < 8; ++j) {
        // If any bit is set in the previous level, set this
        // level's bit. (Max pooling.)
        bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
    }

    uint32_t x = tcnn::morton3D_invert(i>>0) + grid_size/8;
    uint32_t y = tcnn::morton3D_invert(i>>1) + grid_size/8;
    uint32_t z = tcnn::morton3D_invert(i>>2) + grid_size/8;

    next_level[tcnn::morton3D(x, y, z)] |= bits;
}

 void NeuralRadianceField::update_density_grid_mean_and_bitfield(cudaStream_t stream) {
    const uint32_t n_elements = grid_volume();

    size_t size_including_mips = grid_mip_offset(num_cascades) / 8;
    density_grid_bitfield.enlarge(size_including_mips);
    density_grid_mean.enlarge(tcnn::reduce_sum_workspace_size(n_elements));

    CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_mean.data(), 0, sizeof(float), stream));
    tcnn::reduce_sum(density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, density_grid_mean.data(), n_elements, stream);

    tcnn::linear_kernel(nerf_grid_to_bitfield, 0, stream,
        n_elements/8 * num_cascades,
        n_elements/8 * (max_cascade + 1),
        density_grid.data(),
        density_grid_bitfield.data(),
        density_grid_mean.data(),
        min_optical_thickness
    );

    for (uint32_t level = 1; level < num_cascades; ++level) {
        tcnn::linear_kernel(nerf_bitfield_max_pool, 0, stream,
            n_elements/64,
            grid_size,
            get_density_grid_bitfield_mip(level-1),
            get_density_grid_bitfield_mip(level)
        );
    }
}

NGP_NAMESPACE_END
