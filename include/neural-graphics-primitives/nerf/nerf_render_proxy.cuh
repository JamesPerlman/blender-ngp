#pragma once

#include <filesystem/path.h>
#include <string>

#include <Eigen/Dense>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/nerf_descriptor.cuh>
#include <neural-graphics-primitives/nerf/neural_radiance_field.cuh>
#include <neural-graphics-primitives/nerf/render_modifiers.cuh>

NGP_NAMESPACE_BEGIN

// NerfRenderProxy is a reference to a neural radiance field and can be rendered multiple times in the same scene
struct NerfRenderProxy {
	NeuralRadianceField& field;
	BoundingBox aabb;
	Eigen::Matrix4f transform;
	RenderModifiers modifiers;
	Eigen::Matrix4f itransform;

	NerfRenderProxy(
		const NerfDescriptor& descriptor,
		NeuralRadianceField& nerf,
		const RenderModifiersDescriptor& global_modifiers
	)
		: field(nerf)
		, aabb(descriptor.aabb)
		, transform(descriptor.transform)
		, itransform(descriptor.transform.inverse())
		, modifiers(descriptor.modifiers, global_modifiers, itransform)
	{};
};

NGP_NAMESPACE_END
