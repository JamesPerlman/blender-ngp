#pragma once

#include <Eigen/Dense>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf.h>

NGP_NAMESPACE_BEGIN

struct NerfGlobalRay {
	Eigen::Vector3f origin;
	Eigen::Vector3f dir;
	Eigen::Array4f rgba;
	uint32_t idx;
	float depth;
	bool alive;
};

struct NerfProxyRay {
	Eigen::Vector3f origin;
	Eigen::Vector3f dir;
	float t;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
	float mask_alpha;
};

struct RenderDataWorkspace {
private:
	uint32_t _n_nerfs;
	uint32_t _n_global_rays;
	uint32_t _n_proxy_rays;
	uint32_t _n_input_elements;
	uint32_t _n_output_elements;

	precision_t* network_input;
	float* network_output;

	NerfProxyRay* proxy_rays[2];
	
public:
	NerfGlobalRay* global_rays[2];
	NerfGlobalRay* global_rays_hit;

	tcnn::GPUMemory<uint32_t> hit_counter;
	tcnn::GPUMemoryArena::Allocation scratch_alloc = {};
	tcnn::GPUMemory<uint32_t> alive_counter;
	uint32_t n_rays_initialized = 0;
	uint32_t n_rays_alive = 0;
	uint32_t min_steps_per_compaction = 1;
	uint32_t max_steps_per_compaction = 8;

	RenderDataWorkspace() : hit_counter(1), alive_counter(1) {};

	uint32_t get_n_pixels_padded(uint32_t n_pixels) const {
		return tcnn::next_multiple(size_t(n_pixels), size_t(tcnn::batch_size_granularity));
	}

	uint32_t get_n_input_floats() const {
		return sizeof(NerfCoordinate) / 4;
	}

	void enlarge(uint32_t n_nerfs, uint32_t n_pixels, uint32_t padded_output_width, cudaStream_t stream) {

		_n_nerfs = n_nerfs;
		_n_global_rays = get_n_pixels_padded(n_pixels);
		_n_proxy_rays = _n_nerfs * _n_global_rays;
		_n_input_elements = max_steps_per_compaction * _n_proxy_rays * sizeof(NerfCoordinate) / 4;
		_n_output_elements = max_steps_per_compaction * _n_proxy_rays * padded_output_width;

		auto scratch = tcnn::allocate_workspace_and_distribute<
			NerfGlobalRay,
			NerfGlobalRay,
			NerfGlobalRay,
			NerfProxyRay,
			NerfProxyRay,
			precision_t,
			float
		>(
			stream, &scratch_alloc,
			_n_global_rays,
			_n_global_rays,
			_n_global_rays,
			_n_proxy_rays,
			_n_proxy_rays,
			_n_input_elements,
			_n_output_elements
		);

		global_rays[0] = std::get<0>(scratch);
		global_rays[1] = std::get<1>(scratch);
		global_rays_hit = std::get<2>(scratch);

		proxy_rays[0] = std::get<3>(scratch);
		proxy_rays[1] = std::get<4>(scratch);

		network_input = std::get<5>(scratch);
		network_output = std::get<6>(scratch);
	}

	uint32_t get_stride_between_proxy_rays() const {
		return _n_global_rays;
	}

	uint32_t get_n_proxy_rays() const {
		return _n_proxy_rays;
	}

	precision_t* get_nerf_network_input(uint32_t idx) const {
		return &network_input[idx * get_stride_between_proxy_rays()];
	}

	float* get_nerf_network_output(uint32_t idx) const {
		return &network_output[idx * get_stride_between_proxy_rays()];
	}

	NerfProxyRay* get_proxy_rays(uint32_t buffer_idx, uint32_t nerf_idx) const {
		return &proxy_rays[buffer_idx][nerf_idx * get_stride_between_proxy_rays()];
	}

	NerfProxyRay* get_proxy_rays_buffer(uint32_t buffer_idx) const {
		return proxy_rays[buffer_idx];
	}

	void clear() {
		scratch_alloc = {};
	}
};

NGP_NAMESPACE_END
