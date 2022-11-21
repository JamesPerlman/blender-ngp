#pragma once

#include <Eigen/Dense>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/nerf_render_proxy.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

NGP_NAMESPACE_BEGIN

struct NerfPropsGPU {

	tcnn::GPUMemory<BoundingBox> aabbs;
	tcnn::GPUMemory<Eigen::Matrix4f> transforms;
	tcnn::GPUMemory<Eigen::Matrix4f> itransforms;

	void copy_from_host(std::vector<NerfRenderProxy>& proxies) {
		uint32_t n_nerfs = proxies.size();

		std::vector<BoundingBox> _aabbs;
		std::vector<Eigen::Matrix4f> _transforms;
		std::vector<Eigen::Matrix4f> _itransforms;

		_aabbs.reserve(n_nerfs);
		_transforms.reserve(n_nerfs);
		_itransforms.reserve(n_nerfs);

		for (NerfRenderProxy& proxy : proxies) {
			proxy.modifiers.copy_from_host();

			_aabbs.emplace_back(proxy.aabb);
			_transforms.emplace_back(proxy.transform);
			_itransforms.emplace_back(proxy.render_aabb_to_local);
		}

		aabbs.resize_and_copy_from_host(_aabbs);
		transforms.resize_and_copy_from_host(_transforms);
		itransforms.resize_and_copy_from_host(_itransforms);
	}

};

NGP_NAMESPACE_END
