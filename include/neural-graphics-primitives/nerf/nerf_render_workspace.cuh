
#pragma once

#include <tiny-cuda-nn/common.h>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/nerf.h>


NGP_NAMESPACE_BEGIN

struct NerfRenderWorkspace {
	RaysNerfSoa rays[2];
	RaysNerfSoa rays_hit;
	precision_t* network_output;
	float* network_input;
	tcnn::GPUMemory<uint32_t> hit_counter;
	tcnn::GPUMemory<uint32_t> alive_counter;
	uint32_t n_rays_initialized = 0;
	uint32_t n_rays_alive = 0;
	tcnn::GPUMemoryArena::Allocation scratch_alloc = {};

	uint32_t min_steps_inbetween_compaction = 1;
	uint32_t max_steps_inbetween_compaction = 8;

	NerfRenderWorkspace() : hit_counter(1), alive_counter(1) {};

	void enlarge(size_t n_elements, size_t n_extra_dims, size_t padded_output_width, cudaStream_t stream);

	void clear() {
		scratch_alloc = {};
	}
};

NGP_NAMESPACE_END
