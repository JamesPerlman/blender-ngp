/** @file   render_data.cuh
 * 
 *  @author James Perlman, avid instant-ngp fan
 */

#pragma once

#include <Eigen/Dense>

#include <neural-graphics-primitives/camera_models.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/nerf_descriptor.cuh>
#include <neural-graphics-primitives/nerf/render_modifiers_descriptor.cuh>

NGP_NAMESPACE_BEGIN

struct RenderOutputProperties {
    Eigen::Vector2i resolution;
    DownsampleInfo ds;
    uint32_t spp;
    EColorSpace color_space = EColorSpace::Linear;
	ETonemapCurve tonemap_curve = ETonemapCurve::Identity;
    float exposure = 0.f;
    Eigen::Vector4f background_color = Eigen::Vector4f(0.f, 0.f, 0.f, 0.f);
    bool flip_y;

    Eigen::Vector2f get_screen_center() const {
        return Eigen::Vector2f(0.5f, 0.5f);
    }

    RenderOutputProperties(
        const Eigen::Vector2i& resolution,
        const DownsampleInfo& ds,
        uint32_t spp,
        EColorSpace color_space,
        ETonemapCurve tonemap_curve,
        float exposure,
        const Eigen::Vector4f& background_color,
        bool flip_y
    )
        : resolution(resolution)
        , ds(ds)
        , spp(spp)
        , color_space(color_space)
        , tonemap_curve(tonemap_curve)
        , exposure(exposure)
        , background_color(background_color)
        , flip_y(flip_y)
    {};

    RenderOutputProperties() = default;
};

// TODO: abstract these into Camera structs somehow
struct RenderCameraProperties {
    Eigen::Matrix<float, 3, 4> transform;
	ECameraModel model = ECameraModel::Perspective;
    // TODO: refactor this garbage so the camera is responsible for its own ray casting
    float focal_length;
	SphericalQuadrilateral spherical_quadrilateral = SphericalQuadrilateral::Zero();
	QuadrilateralHexahedron quadrilateral_hexahedron = QuadrilateralHexahedron::Zero();
    float near_distance;
    float aperture_size;
    float focus_z;

    RenderCameraProperties(
        const Eigen::Matrix<float, 3, 4>& transform,
        ECameraModel model,
        float focal_length,
        float near_distance,
        float aperture_size,
        float focus_z,
        const SphericalQuadrilateral& spherical_quadrilateral,
        const QuadrilateralHexahedron& quadrilateral_hexahedron
    )
        : transform(transform)
        , model(model)
        , focal_length(focal_length)
        , near_distance(near_distance)
        , aperture_size(aperture_size)
        , focus_z(focus_z)
        , spherical_quadrilateral(spherical_quadrilateral)
        , quadrilateral_hexahedron(quadrilateral_hexahedron)
    {};
    
    RenderCameraProperties() = default;

    // equality operator
    bool operator==(const RenderCameraProperties& other) const {
        return transform == other.transform
            && model == other.model
            && focal_length == other.focal_length
            && near_distance == other.near_distance
            && aperture_size == other.aperture_size
            && focus_z == other.focus_z
            && spherical_quadrilateral == other.spherical_quadrilateral
            && quadrilateral_hexahedron == other.quadrilateral_hexahedron;
    }

    bool operator!=(const RenderCameraProperties& other) const {
        return !(*this == other);
    }
};

struct RenderRequest {
    RenderOutputProperties output;
    RenderCameraProperties camera;
    RenderModifiersDescriptor modifiers;
    std::vector<NerfDescriptor> nerfs;
    BoundingBox aabb;

    RenderRequest(
        const RenderOutputProperties& output,
        const RenderCameraProperties& camera,
        const RenderModifiersDescriptor& modifiers,
        const std::vector<NerfDescriptor>& nerfs,
        const BoundingBox& aabb
    )
        : output(output)
        , camera(camera)
        , modifiers(modifiers)
        , nerfs(nerfs)
        , aabb(aabb)
        {};
};

NGP_NAMESPACE_END
