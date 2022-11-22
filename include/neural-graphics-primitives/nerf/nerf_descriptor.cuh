#pragma once

#include <filesystem/path.h>
#include <string>

#include <Eigen/Dense>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/render_modifiers_descriptor.cuh>

NGP_NAMESPACE_BEGIN

// NerfDescriptor is passed in externally and is just a lightweight reference to a NeuralRadianceField
struct NerfDescriptor {
    ::filesystem::path snapshot_path;
    BoundingBox aabb;
    Eigen::Matrix4f transform;
    RenderModifiersDescriptor modifiers;

    NerfDescriptor(
        const std::string& snapshot_path_str,
        const BoundingBox& aabb,
        const Eigen::Matrix4f& transform,
        const RenderModifiersDescriptor& modifiers
    )
        : snapshot_path(snapshot_path_str)
        , aabb(aabb)
        , modifiers(modifiers)
        , transform(transform)
    {};
};

NGP_NAMESPACE_END