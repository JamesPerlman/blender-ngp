

#include <Eigen/Dense>

#include <tiny-cuda-nn/common.h>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/nerf_render_workspace.cuh>

NGP_NAMESPACE_BEGIN

void NerfRenderWorkspace::enlarge(size_t n_elements, size_t n_extra_dims, size_t padded_output_width, cudaStream_t stream) {
	n_elements = tcnn::next_multiple(n_elements, size_t(tcnn::batch_size_granularity));
	size_t num_floats = sizeof(NerfCoordinate) / 4 + n_extra_dims;

	auto scratch = tcnn::allocate_workspace_and_distribute<
		Eigen::Array4f, float, NerfPayload, // m_rays[0]
		Eigen::Array4f, float, NerfPayload, // m_rays[1]
		Eigen::Array4f, float, NerfPayload, // m_rays_hit
		tcnn::network_precision_t,
		float
	>(
		stream, &scratch_alloc,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements * max_steps_inbetween_compaction * padded_output_width,
		n_elements * max_steps_inbetween_compaction * num_floats
		);

	rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
	rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
	rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

	network_output = std::get<9>(scratch);
	network_input = std::get<10>(scratch);
}

NGP_NAMESPACE_END
