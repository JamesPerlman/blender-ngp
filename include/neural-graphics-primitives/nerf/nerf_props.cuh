#pragma once

#include <Eigen/Dense>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/nerf_render_proxy.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

NGP_NAMESPACE_BEGIN

struct NerfProps {

	Eigen::Matrix4f transform;
	Eigen::Matrix4f itransform;
	uint8_t* density_grid_bitfield;
	uint32_t grid_size;
	uint32_t grid_volume;
	BoundingBox render_aabb;
	BoundingBox train_aabb;
	Mask3D* masks;
	uint32_t n_masks;
	float cone_angle;
	float min_cone_stepsize;
	float max_cone_stepsize;
	uint32_t nerf_cascades;

	NerfProps(const NerfRenderProxy& nerf)
		: transform(nerf.transform)
		, itransform(nerf.itransform)
		, density_grid_bitfield(nerf.field.density_grid_bitfield.data())
		, grid_size(nerf.field.grid_size)
		, grid_volume(nerf.field.get_grid_volume())
		, render_aabb(nerf.aabb)
		, train_aabb(nerf.field.train_aabb)
		, masks(nerf.modifiers.masks.data())
		, n_masks(nerf.modifiers.masks.size())
		, cone_angle(nerf.field.cone_angle_constant)
		, min_cone_stepsize(nerf.field.get_min_cone_step_size())
		, max_cone_stepsize(nerf.field.get_max_cone_step_size())
		, nerf_cascades(nerf.field.num_cascades)
	{};

};

NGP_NAMESPACE_END
