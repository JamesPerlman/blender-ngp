
#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/render_data.cuh>
#include <neural-graphics-primitives/nerf/render_request.cuh>
#include <neural-graphics-primitives/render_buffer.h>

NGP_NAMESPACE_BEGIN

struct NerfRenderer {
private:
	RenderData m_render_data;

public:

	void render(
		CudaRenderBuffer& render_buffer,
		RenderRequest& render_request,
		cudaStream_t stream
	);

	void init_rays_from_camera(
		Eigen::Array4f* frame_buffer,
		float* depth_buffer,
		RenderData& render_data,
		cudaStream_t stream
	);

	// returns number of rays hit
	uint32_t march_rays_and_accumulate_colors(
		RenderData& render_data,
		cudaStream_t stream
	);

};

NGP_NAMESPACE_END
