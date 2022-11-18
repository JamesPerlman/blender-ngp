#pragma once

#include <filesystem/path.h>
#include <string>

#include <Eigen/Dense>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/nerf_descriptor.cuh>
#include <neural-graphics-primitives/nerf/nerf_render_workspace.cuh>
#include <neural-graphics-primitives/nerf/neural_radiance_field.cuh>
#include <neural-graphics-primitives/nerf/render_modifiers.cuh>

NGP_NAMESPACE_BEGIN

// NerfRenderProxy is a reference to a neural radiance field and can be rendered multiple times in the same scene
struct NerfRenderProxy {
	NeuralRadianceField& field;
	BoundingBox aabb;
	Eigen::Matrix4f transform;
	RenderModifiers modifiers;
	Eigen::Matrix4f render_aabb_to_local;
	NerfRenderWorkspace workspace;

	NerfRenderProxy(const NerfDescriptor& descriptor, NeuralRadianceField& nerf)
		: field(nerf)
		, aabb(descriptor.aabb)
		, transform(descriptor.transform)
		, modifiers(descriptor.modifiers)
		, render_aabb_to_local(descriptor.transform.inverse())
		, workspace()
	{};
};

NGP_NAMESPACE_END
