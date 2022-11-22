#pragma once

#include <Eigen/Dense>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/nerf_render_proxy.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

NGP_NAMESPACE_BEGIN

struct NerfPropsGPU {

	tcnn::GPUMemory<Eigen::Matrix4f> transforms;
	tcnn::GPUMemory<Eigen::Matrix4f> itransforms;
	tcnn::GPUMemory<uint8_t*> density_grid_bitfield_ptrs;
	tcnn::GPUMemory<uint32_t> grid_sizes;
	tcnn::GPUMemory<BoundingBox> render_aabbs;
	tcnn::GPUMemory<BoundingBox> train_aabbs;
	tcnn::GPUMemory<Mask3D*> local_mask_ptrs;
	tcnn::GPUMemory<uint32_t> local_mask_sizes;
	tcnn::GPUMemory<float> cone_angles;

	void copy_from_host(std::vector<NerfRenderProxy>& proxies) {
		uint32_t n_nerfs = proxies.size();

		std::vector<Eigen::Matrix4f> _transforms;
		std::vector<Eigen::Matrix4f> _itransforms;
		std::vector<uint8_t*> _density_grid_bitfield_ptrs;
		std::vector<uint32_t> _grid_sizes;
		std::vector<BoundingBox> _render_aabbs;
		std::vector<BoundingBox> _train_aabbs;
		std::vector<Mask3D*> _local_mask_ptrs;
		std::vector<uint32_t> _local_mask_sizes;
		std::vector<float> _cone_angles;

		_transforms.reserve(n_nerfs);
		_itransforms.reserve(n_nerfs);
		_density_grid_bitfield_ptrs.reserve(n_nerfs);
		_grid_sizes.reserve(n_nerfs);
		_render_aabbs.reserve(n_nerfs);
		_local_mask_ptrs.reserve(n_nerfs);
		_local_mask_sizes.reserve(n_nerfs);
		_cone_angles.reserve(n_nerfs);

		for (NerfRenderProxy& proxy : proxies) {
			proxy.modifiers.copy_from_host();

			_render_aabbs.emplace_back(proxy.aabb);
			_train_aabbs.emplace_back(proxy.field.train_aabb);
			_transforms.emplace_back(proxy.transform);
			_itransforms.emplace_back(proxy.render_aabb_to_local);
			_density_grid_bitfield_ptrs.emplace_back(proxy.field.density_grid_bitfield.data());
			_grid_sizes.emplace_back(proxy.field.grid_size);
			_local_mask_ptrs.emplace_back(proxy.modifiers.masks.data());
			_local_mask_sizes.emplace_back(proxy.modifiers.masks.size());
			_cone_angles.emplace_back(proxy.field.cone_angle_constant);
		}

		render_aabbs.resize_and_copy_from_host(_render_aabbs);
		transforms.resize_and_copy_from_host(_transforms);
		itransforms.resize_and_copy_from_host(_itransforms);
		density_grid_bitfield_ptrs.resize_and_copy_from_host(_density_grid_bitfield_ptrs);
		grid_sizes.resize_and_copy_from_host(_grid_sizes);
		local_mask_ptrs.resize_and_copy_from_host(_local_mask_ptrs);
		local_mask_sizes.resize_and_copy_from_host(_local_mask_sizes);
		cone_angles.resize_and_copy_from_host(_cone_angles);
	}

};

NGP_NAMESPACE_END
