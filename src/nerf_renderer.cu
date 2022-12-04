#include <neural-graphics-primitives/camera_models.cuh>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/mask_3D.cuh>
#include <neural-graphics-primitives/nerf/nerf_utils.cuh>
#include <neural-graphics-primitives/nerf/render_data.cuh>
#include <neural-graphics-primitives/nerf/nerf_props.cuh>
#include <neural-graphics-primitives/nerf/nerf_renderer.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

__global__ void init_global_rays_kernel(
	const uint32_t sample_index,
	NerfGlobalRay* __restrict__ rays,
	float* __restrict__ depthbuffer,
	const DownsampleInfo ds,
	const RenderCameraProperties camera
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
	uint32_t idx = x + ds.scaled_res.x() * y;

	x *= ds.skip.x();
	y *= ds.skip.y();

	if (x >= ds.max_res.x() || y >= ds.max_res.y()) {
		return;
	}

	NerfGlobalRay& ray = rays[idx];

	// TODO: pixel_to_ray also immediately computes u,v for the pixel, so this is somewhat redundant
	float u = (x + 0.5f) * (1.f / ds.max_res.x());
	float v = (y + 0.5f) * (1.f / ds.max_res.y());

	const Eigen::Vector2i pixel = { x, y };

	Ray r = { {0.0f, 0.0f, 0.f}, {0.f, 0.f, 1.f} };

	switch (camera.model) {
	case ECameraModel::Perspective:
		r = perspective_pixel_to_ray(
			sample_index,
			pixel,
			ds.max_res,
			Eigen::Vector2f(camera.focal_length, camera.focal_length), // TODO: fx and fy
			camera.transform,
			camera.near_distance,
			camera.aperture_size,
			camera.focus_z
		);
		break;
	case ECameraModel::SphericalQuadrilateral:
		r = spherical_quadrilateral_pixel_to_ray(
			sample_index,
			pixel,
			ds.max_res,
			camera.transform,
			camera.spherical_quadrilateral,
			camera.near_distance,
			camera.focus_z,
			camera.aperture_size
		);
		break;
	case ECameraModel::QuadrilateralHexahedron:
		r = quadrilateral_hexahedron_pixel_to_ray(
			sample_index,
			pixel,
			ds.max_res,
			camera.transform,
			camera.quadrilateral_hexahedron,
			camera.near_distance,
			camera.focus_z,
			camera.aperture_size
		);
		break;
	}

	depthbuffer[idx] = 1e10f;

	ray.origin = r.o;
	ray.dir = r.d.normalized();
	ray.idx = idx;
	ray.alive = true;
	ray.depth = 0.0f;
	ray.rgba = Array4f::Zero();
}

__global__ void init_proxy_rays_kernel(
	uint32_t n_elements,
	const NerfGlobalRay* __restrict__ global_rays,
	NerfProxyRay* proxy_rays,
	const NerfProps* __restrict__ nerf_props
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_elements) {
		return;
	}

	const NerfGlobalRay& global_ray = global_rays[i];
	NerfProxyRay& proxy_ray = proxy_rays[global_ray.idx];

	if (!global_ray.alive) {
		proxy_ray.alive = false;
		proxy_ray.origin = Vector3f::Zero();
		return;
	}

	Vector3f origin = nerf_props->render_aabb.localized_point(nerf_props->itransform, global_ray.origin);
	proxy_ray.dir = nerf_props->render_aabb.localized_direction(nerf_props->itransform, global_ray.dir);
	const Ray local_r = { origin, proxy_ray.dir };

	// TODO: scene transform
	float t = fmaxf(nerf_props->render_aabb.ray_intersect(origin, proxy_ray.dir).x(), 0.0f) + 1e-5f;

	if (!nerf_props->render_aabb.contains(origin + proxy_ray.dir * t)) {
		proxy_ray.alive = false;
		return;
	}

	bool ray_intersects_any_mask = nerf_props->n_masks == 0;

	// test local masks
	if (!ray_intersects_any_mask) {
		for (uint32_t k = 0; k < nerf_props->n_masks; ++k) {
			if (nerf_props->masks[k].intersects_ray(local_r)) {
				ray_intersects_any_mask = true;
				break;
			}
		}
	}

	proxy_ray.active = true;
	proxy_ray.alive = ray_intersects_any_mask;
	proxy_ray.idx = global_ray.idx;
	proxy_ray.t = 0.0f;
	proxy_ray.n_steps = 0;
	proxy_ray.origin = origin + t * proxy_ray.dir;
}


__device__ bool hit_test_and_march(
	const Eigen::Vector3f& proxy_origin,
	const Eigen::Vector3f& proxy_dir,
	const Eigen::Vector3f& proxy_idir,
	const float proxy_t,
	const NerfProps* __restrict__ nerf_props,
	float* t_out,
	float* dt_out
) {
	Eigen::Vector3f pos;
	float t = proxy_t;
	float dt = 0.0f;

	float prev_t = t;
	while (1) {
		// TODO: Distortion fields
		pos = proxy_origin + proxy_dir * t;
		if (!nerf_props->render_aabb.contains(pos)) {
			if (t_out)
				*t_out = prev_t;
			if (dt_out)
				*dt_out = dt;

			return false;
		}

		dt = get_dt(t, nerf_props->cone_angle, nerf_props->min_cone_stepsize, nerf_props->max_cone_stepsize);
		uint32_t mip = max(0, get_mip_from_dt(dt, pos, nerf_props->grid_size, nerf_props->nerf_cascades - 1));
		if (!nerf_props->density_grid_bitfield) {
			break;
		}

		// test density grid first, then test masks
		if (get_is_density_grid_occupied_at(pos, nerf_props->density_grid_bitfield, mip, nerf_props->grid_size, nerf_props->grid_volume)) {
			bool hits_any_mask = false;
			break;
			// test local masks
			for (uint32_t k = 0; k < nerf_props->n_masks; ++k) {
				if (nerf_props->masks[k].contains(pos)) {
					hits_any_mask = true;
					break;
				}
			}

			if (hits_any_mask) {
				break;
			}
		}

		uint32_t res = nerf_props->grid_size >> mip;
		prev_t = t;
		t = get_t_advanced_to_next_voxel(t, nerf_props->cone_angle, pos, proxy_dir, proxy_idir, res, nerf_props->min_cone_stepsize, nerf_props->max_cone_stepsize);
	} // while

	if (t_out)
		*t_out = t;
	if (dt_out)
		*dt_out = dt;

	return true;
}


__global__ void march_proxy_rays_init_kernel(
	const uint32_t n_elements,
	const uint32_t sample_index,
	const Vector3f camera_fwd,
	NerfProxyRay* __restrict__ rays,
	const NerfProps* __restrict__ nerf_props
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfProxyRay& proxy_ray = rays[i];

	if (!proxy_ray.alive) {
		return;
	}

	Vector3f origin = proxy_ray.origin;
	Vector3f dir = proxy_ray.dir;
	Vector3f idir = dir.cwiseInverse();

	float t = proxy_ray.t;
	float dt = get_dt(t, nerf_props->cone_angle, nerf_props->min_cone_stepsize, nerf_props->max_cone_stepsize);
	t += ld_random_val(sample_index, i * 786433) * dt;

	proxy_ray.alive = hit_test_and_march(origin, dir, idir, t, nerf_props, &t, &dt);

	proxy_ray.t = t;
}

__global__ void compact_rays_kernel(
	uint32_t n_elements,
	NerfGlobalRay* global_src_rays,
	NerfGlobalRay* global_dst_rays,
	NerfProxyRay* proxy_src_rays,
	NerfProxyRay* proxy_dst_rays,
	uint32_t n_nerfs,
	uint32_t proxy_rays_stride_between_nerfs,
	NerfGlobalRay* global_final_rays,
	uint32_t* global_alive_counter,
	uint32_t* global_final_counter
) {

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfGlobalRay& global_src_ray = global_src_rays[i];

	if (global_src_ray.alive) {
		uint32_t idx = atomicAdd(global_alive_counter, 1);
		global_dst_rays[idx] = global_src_ray;
		for (uint32_t n = 0; n < n_nerfs; ++n) {
			uint32_t offset = n * proxy_rays_stride_between_nerfs;
			proxy_dst_rays[idx + offset] = proxy_src_rays[i + offset];
		}
	}
	else if (global_src_ray.rgba.w() > 0.001f) {
		uint32_t idx = atomicAdd(global_final_counter, 1);
		global_final_rays[idx] = global_src_ray;
	}
}

// march rays until the moment they become active - does not do marching per step, only marches each ray once until it hits its next sample point

__global__ void march_active_rays(
	const uint32_t n_rays_alive,
	const uint32_t n_nerfs,
	const Vector3f camera_fwd,
	const NerfGlobalRay* __restrict__ global_rays,
	NerfProxyRay* proxy_rays,
	const uint32_t proxy_rays_stride_between_nerfs,
	const NerfProps* __restrict__ nerf_props
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_rays_alive) {
		return;
	};

	const NerfGlobalRay& global_ray = global_rays[i];
	if (!global_ray.alive) {
		return;
	}

	for (uint32_t n = 0; n < n_nerfs; ++n) {
		uint32_t proxy_ray_idx = i + n * proxy_rays_stride_between_nerfs;
		NerfProxyRay& proxy_ray = proxy_rays[proxy_ray_idx];
		if (!proxy_ray.alive || !proxy_ray.active)
			continue;

		Vector3f origin = proxy_ray.origin;
		Vector3f dir = proxy_ray.dir;
		Vector3f idir = dir.cwiseInverse();
		float t = proxy_ray.t;
		proxy_ray.alive = hit_test_and_march(
			origin,
			dir,
			idir,
			t,
			nerf_props + n,
			&t,
			nullptr
		);
		proxy_ray.t = t;
	}
}

__global__ void march_proxy_rays_and_generate_next_network_inputs(
	const uint32_t n_elements,
	const NerfGlobalRay* __restrict__ global_rays,
	NerfProxyRay* proxy_rays,
	NerfCoordinate* network_input,
	uint32_t n_steps,
	Vector3f camera_fwd,
	const NerfProps* __restrict__ nerf_props
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_elements) {
		return;
	};

	const NerfGlobalRay& global_ray = global_rays[i];

	if (!global_ray.alive) {
		return;
	}

	NerfProxyRay& proxy_ray = proxy_rays[i];

	if (!proxy_ray.active) {
		return;
	}

	Vector3f origin = proxy_ray.origin;
	Vector3f dir = proxy_ray.dir;
	Vector3f idir = dir.cwiseInverse();

	float t = proxy_ray.t;
	float dt = get_dt(t, nerf_props->cone_angle, nerf_props->min_cone_stepsize, nerf_props->max_cone_stepsize);


	for (uint32_t j = 0; j < n_steps; ++j) {
		Vector3f pos = origin + dir * t;
		network_input[i + j * n_elements].set_with_optional_extra_dims(
			get_warped_pos(pos, nerf_props->train_aabb),
			get_warped_dir(dir),
			get_warped_dt(dt, nerf_props->min_cone_stepsize, nerf_props->nerf_cascades),
			nullptr,
			sizeof(NerfCoordinate)
		); // XXXCONE

		bool ray_alive = hit_test_and_march(origin, dir, idir, t, nerf_props, &t, &dt);

		if (!ray_alive) {
			proxy_ray.n_steps = j;
			return;
		}
		t += dt;
	} // for

	proxy_ray.t = t;
	proxy_ray.n_steps = n_steps;
}

// cull rays
__global__ void cull_global_rays_and_set_proxy_rays_active_kernel(
	const uint32_t n_rays_alive,
	const uint32_t n_nerfs,
	NerfGlobalRay* global_rays,
	NerfProxyRay* proxy_rays,
	const uint32_t proxy_rays_stride_between_nerfs,
	const Vector3f cam_pos,
	const NerfProps* __restrict__ nerf_props
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_rays_alive) {
		return;
	}

	NerfGlobalRay& global_ray = global_rays[i];

	if (!global_ray.alive) {
		return;
	}

	float min_d2 = 0.0f;
	int32_t active_idx = -1;
	uint32_t n_proxy_alive = 0;
	for (uint32_t n = 0; n < n_nerfs; ++n) {
		const uint32_t proxy_ray_idx = i + n * proxy_rays_stride_between_nerfs;
		NerfProxyRay& proxy_ray = proxy_rays[proxy_ray_idx];
		if (!proxy_ray.alive) {
			continue;
		}
		++n_proxy_alive;

		Vector3f p = proxy_ray.origin + proxy_ray.dir * proxy_ray.t;
		p = (nerf_props[n].transform * p.homogeneous()).head<3>();
		float d2 = (p - cam_pos).squaredNorm();

		if (d2 < min_d2 || active_idx == -1) {
			min_d2 = d2;
			active_idx = proxy_ray_idx;
		}

		proxy_ray.active = false;
	}

	// turn back on the ray with the lowest t index
	if (active_idx >= 0) {
		proxy_rays[active_idx].active = true;
	}

	if (n_proxy_alive == 0) {
		global_ray.alive = false;
	}
}


__global__ void composite_proxy_ray_colors_kernel(
	const uint32_t n_global_rays,
	const uint32_t current_step,
	NerfGlobalRay* global_rays,
	NerfProxyRay* proxy_rays,
	Matrix<float, 3, 4> camera_matrix,
	const NerfCoordinate* __restrict__ network_input,
	const tcnn::network_precision_t* __restrict__ network_output,
	const uint32_t n_network_elements,
	uint32_t n_steps,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	float min_transmittance,
	const NerfProps* __restrict__ nerf_props
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_global_rays) return;

	NerfGlobalRay& global_ray = global_rays[i];

	if (!global_ray.alive) {
		return;
	}

	NerfProxyRay& proxy_ray = proxy_rays[i];

	if (!proxy_ray.alive || !proxy_ray.active) {
		return;
	}

	Array4f local_rgba = global_ray.rgba;


	Vector3f cam_fwd = camera_matrix.col(2);

	// Composite in the last n steps
	uint32_t actual_n_steps = proxy_ray.n_steps;
	uint32_t j = 0;


	for (; j < actual_n_steps; ++j) {
		tcnn::vector_t<tcnn::network_precision_t, 4> proxy_network_output;
		proxy_network_output[0] = network_output[i + j * n_global_rays + 0 * n_network_elements];
		proxy_network_output[1] = network_output[i + j * n_global_rays + 1 * n_network_elements];
		proxy_network_output[2] = network_output[i + j * n_global_rays + 2 * n_network_elements];
		proxy_network_output[3] = network_output[i + j * n_global_rays + 3 * n_network_elements];
		const NerfCoordinate& input = network_input[i + j * n_global_rays];

		Vector3f warped_pos = input.pos.p;
		Vector3f pos = get_unwarped_pos(warped_pos, nerf_props->train_aabb);

		float T = 1.f - local_rgba.w();
		float dt = get_unwarped_dt(input.dt, nerf_props->min_cone_stepsize, nerf_props->nerf_cascades);
		float alpha = 1.f - __expf(-get_network_density(float(proxy_network_output[3]), density_activation) * dt);
		float weight = alpha * T;

		Array3f rgb = get_network_rgb(proxy_network_output, rgb_activation);

		float mask_weight = 1.f;

		for (uint32_t k = 0; k < nerf_props->n_masks; ++k) {
			float mask_alpha = nerf_props->masks[k].sample(pos);
			mask_weight = tcnn::clamp(mask_weight + mask_alpha, 0.0f, 1.0f);
		}
		weight *= mask_weight;
		weight *= nerf_props->opacity;

		local_rgba.head<3>() += rgb * weight;
		local_rgba.w() += weight;

		if (local_rgba.w() > (1.0f - min_transmittance)) {
			local_rgba /= local_rgba.w();
			break;
		}

	}

	// we broke out of the step loop early, ray must have terminated
	if (j < n_steps) {
		proxy_ray.alive = false;
		proxy_ray.n_steps = j + current_step;
	}

	global_ray.rgba = local_rgba;
	// global_ray.depth = global_depth;
}

__global__ void shade_buffer_with_rays_kernel(
	const uint32_t n_rays,
	const NerfGlobalRay* rays,
	bool train_in_linear_colors,
	Array4f* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	const DownsampleInfo ds,
	const bool flip_y
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	const NerfGlobalRay& ray = rays[i];

	// get actual x and y coordinates, accounting for downsampling
	uint32_t x = ds.skip.x() * (ray.idx % (uint32_t)ds.scaled_res.x());
	uint32_t y = ds.skip.y() * (ray.idx / (uint32_t)ds.scaled_res.x());

	if (flip_y) {
		y = ds.max_res.y() - y - 1;
	}

	Array4f tmp = ray.rgba;

	if (!train_in_linear_colors) {
		// Accumulate in linear colors
		tmp.head<3>() = srgb_to_linear(tmp.head<3>());
	}

	for (uint32_t u = 0; u < ds.skip.x(); ++u) {
		for (uint32_t v = 0; v < ds.skip.y(); ++v) {

			uint32_t idx = (x + u) + (y + v) * ds.max_res.x();

			if (idx >= ds.max_pixels) {
				continue;
			}

			frame_buffer[idx] = tmp + frame_buffer[idx] * (1.0f - tmp.w());
			if (tmp.w() > 0.2f) {
				depth_buffer[idx] = ray.depth;
			}
		}
	}
}

void NerfRenderer::render(
	CudaRenderBuffer& render_buffer,
	RenderRequest& render_request,
	cudaStream_t stream
) {
	if (render_request.nerfs.size() == 0) {
		return;
	}

	m_render_data.update(render_request);
	m_render_data.copy_from_host();

	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	std::vector<NerfRenderProxy>& nerfs = m_render_data.get_renderables();

	ScopeGuard tmp_memory_guard{ [&]() {
		m_render_data.workspace.clear();
	} };

	// allocate workspace for nerfs
	size_t n_pixels = (size_t)(m_render_data.output.ds.scaled_pixels);

	// Make sure we have enough memory reserved to render at the requested resolution
	// assumption: all nerfs have the same network padded_output_width
	m_render_data.workspace.enlarge(nerfs.size(), n_pixels, nerfs[0].field.network->padded_output_width(), stream);

	init_rays_from_camera(render_buffer.frame_buffer(), render_buffer.depth_buffer(), m_render_data, stream);

	uint32_t n_hit = march_rays_and_accumulate_colors(m_render_data, stream);

	NerfGlobalRay* rays_hit = m_render_data.workspace.global_rays_hit;

	linear_kernel(shade_buffer_with_rays_kernel, 0, stream,
		n_hit,
		rays_hit,
		false, //m_nerf.training.linear_colors,
		render_buffer.frame_buffer(),
		render_buffer.depth_buffer(),
		m_render_data.output.ds,
		m_render_data.output.flip_y
	);

}

void NerfRenderer::init_rays_from_camera(
	Eigen::Array4f* frame_buffer,
	float* depth_buffer,
	RenderData& render_data,
	cudaStream_t stream
) {
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = {
		div_round_up((unsigned int)render_data.output.ds.scaled_res.x(), threads.x),
		div_round_up((unsigned int)render_data.output.ds.scaled_res.y(), threads.y),
		1
	};

	init_global_rays_kernel<<<blocks, threads, 0, stream>>> (
		0, // todo: sample index
		render_data.workspace.global_rays[0],
		depth_buffer,
		render_data.output.ds,
		render_data.camera
	);

	render_data.workspace.n_rays_initialized = render_data.output.ds.scaled_pixels;

	std::vector<NerfRenderProxy>& nerfs = render_data.get_renderables();
	for (uint32_t i = 0; i < nerfs.size(); ++i) {
		NerfRenderProxy& nerf = nerfs[i];

		NerfProxyRay* proxy_rays = render_data.workspace.get_proxy_rays(0, i);

		linear_kernel(init_proxy_rays_kernel, 0, stream,
			render_data.workspace.n_rays_initialized,
			render_data.workspace.global_rays[0],
			proxy_rays,
			render_data.nerf_props.data() + i
		);
		/*

		linear_kernel(march_proxy_rays_init_kernel, 0, stream,
			render_data.workspace.n_rays_initialized,
			0, // todo: sample_index
			render_data.camera.transform.col(2),
			proxy_rays,
			render_data.nerf_props.data() + i
		);*/
	}
	// TODO: cull global rays?

}


// returns number of rays hit
uint32_t NerfRenderer::march_rays_and_accumulate_colors(
	RenderData& render_data,
	cudaStream_t stream
) {

	render_data.workspace.n_rays_alive = render_data.workspace.n_rays_initialized;

	CUDA_CHECK_THROW(cudaMemsetAsync(render_data.workspace.hit_counter.data(), 0, sizeof(uint32_t), stream));

	std::vector<NerfRenderProxy>& nerfs = render_data.get_renderables();
	uint32_t n_nerfs = nerfs.size();

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;

	while (i < 10000) {

		uint32_t rays_tmp_index = double_buffer_index % 2;
		uint32_t rays_current_index = (double_buffer_index + 1) % 2;

		NerfGlobalRay* global_rays_tmp = render_data.workspace.global_rays[rays_tmp_index];
		NerfGlobalRay* global_rays_current = render_data.workspace.global_rays[rays_current_index];

		NerfProxyRay* proxy_rays_tmp = render_data.workspace.get_proxy_rays_buffer(rays_tmp_index);
		NerfProxyRay* proxy_rays_current = render_data.workspace.get_proxy_rays_buffer(rays_current_index);

		++double_buffer_index;
		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(render_data.workspace.alive_counter.data(), 0, sizeof(uint32_t), stream));

			linear_kernel(compact_rays_kernel, 0, stream,
				render_data.workspace.n_rays_alive,
				global_rays_tmp,
				global_rays_current,
				proxy_rays_tmp,
				proxy_rays_current,
				(uint32_t)nerfs.size(),
				render_data.workspace.get_proxy_rays_stride_between_nerfs(),
				render_data.workspace.global_rays_hit,
				render_data.workspace.alive_counter.data(),
				render_data.workspace.hit_counter.data()
			);
			CUDA_CHECK_THROW(cudaMemcpyAsync(&render_data.workspace.n_rays_alive, render_data.workspace.alive_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}

		if (render_data.workspace.n_rays_alive == 0) {
			break;
		}

		// march all active rays across all nerfs

		linear_kernel(march_active_rays, 0, stream,
			render_data.workspace.n_rays_alive,
			n_nerfs,
			render_data.camera.transform.col(2),
			global_rays_current,
			proxy_rays_current,
			render_data.workspace.get_proxy_rays_stride_between_nerfs(),
			render_data.nerf_props.data()
		);

		// filter all rays across all nerfs and select only the alive ones with the lowest t-value to pass into the network
		linear_kernel(cull_global_rays_and_set_proxy_rays_active_kernel, 0, stream,
			render_data.workspace.n_rays_alive,
			n_nerfs,
			global_rays_current,
			proxy_rays_current,
			render_data.workspace.get_proxy_rays_stride_between_nerfs(),
			render_data.camera.transform.col(3),
			render_data.nerf_props.data()
		);

		uint32_t n_steps_between_compaction = tcnn::clamp(render_data.workspace.n_rays_initialized / render_data.workspace.n_rays_alive, render_data.workspace.min_steps_per_compaction, render_data.workspace.max_steps_per_compaction);
		uint32_t n_network_elements = next_multiple(render_data.workspace.n_rays_alive * n_steps_between_compaction, tcnn::batch_size_granularity);

		for (uint32_t j = 0; j < n_nerfs; ++j) {
			NerfRenderProxy& nerf = nerfs[j];

			linear_kernel(march_proxy_rays_and_generate_next_network_inputs, 0, stream,
				render_data.workspace.n_rays_alive,
				global_rays_current,
				render_data.workspace.get_proxy_rays(rays_current_index, j),
				render_data.workspace.get_nerf_network_input(),
				n_steps_between_compaction,
				render_data.camera.transform.col(2),
				render_data.nerf_props.data() + j
			);

			GPUMatrix<float> positions_matrix(
				(float*)render_data.workspace.get_nerf_network_input(),
				sizeof(NerfCoordinate) / sizeof(float),
				n_network_elements
			);

			GPUMatrix<network_precision_t, RM> rgbsigma_matrix(
				(network_precision_t*)render_data.workspace.get_nerf_network_output(),
				nerf.field.network->padded_output_width(),
				n_network_elements
			);

			nerf.field.network->inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

			// composite network outputs as RGBA across all nerfs, accumulating colors in render_data.workspace.global_rays[...].rgba
			linear_kernel(composite_proxy_ray_colors_kernel, 0, stream,
				render_data.workspace.n_rays_alive,
				i,
				global_rays_current,
				render_data.workspace.get_proxy_rays(rays_current_index, j),
				render_data.camera.transform.block<3, 4>(0, 0),
				render_data.workspace.get_nerf_network_input(),
				(network_precision_t*)render_data.workspace.get_nerf_network_output(),
				n_network_elements,
				n_steps_between_compaction,
				nerf.field.rgb_activation,
				nerf.field.density_activation,
				nerf.field.min_transmittance,
				render_data.nerf_props.data() + j
			);
		}


		i += n_steps_between_compaction;
	}

	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, render_data.workspace.hit_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}


NGP_NAMESPACE_END
