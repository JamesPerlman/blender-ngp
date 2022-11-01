/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   camera_path.h
 *  @author James Perlman, avid instant-ngp fan
 */

#pragma once

#include <Eigen/Dense>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/random_val.cuh>

#include <tiny-cuda-nn/common.h>

using namespace Eigen;

NGP_NAMESPACE_BEGIN

enum class EMaskMode {
    Add,
    Subtract,
};

enum class EMaskShape : int {
    Box,
    Cylinder,
    Sphere,
};

// thank you https://iquilezles.org/articles/distfunctions/ for the sdf primitives

// xyz -> yzx

inline NGP_HOST_DEVICE float sdfBox(const Vector3f& p, const Vector3f& b) {
    Vector3f d = p.cwiseAbs() - 0.5f * b;
    return d.cwiseMax(0.0f).norm() + min(max(d.x(), max(d.y(), d.z())), 0.0f);
};

inline NGP_HOST_DEVICE float sdfCylinder(const Vector3f& p, const float& r, const float& h) {
    Vector2f d = Vector2f(Vector2f(p.y(), p.x()).norm(), p.z()).cwiseAbs() - Vector2f(r, 0.5f * h);
    return d.cwiseMax(0.0f).norm() + min(max(d.x(), d.y()), 0.0f);
};

inline NGP_HOST_DEVICE float sdfSphere(const Vector3f& p, const float& r) {
    return p.norm() - r;
};

struct Mask3D {
    EMaskMode mode;
    EMaskShape shape;
    Matrix4f transform;
    Vector4f config;
    float feather;
    float opacity;

    NGP_HOST_DEVICE Mask3D(const EMaskShape& shape, const Matrix4f& transform, const EMaskMode& mode, const Vector4f& config, const float& feather, const float& opacity)
        : mode(mode), shape(shape), transform(transform), config(config), feather(feather), opacity(opacity) {};
    
    NGP_HOST_DEVICE Mask3D() : mode(EMaskMode::Add), shape(EMaskShape::Box), transform(Matrix4f::Identity()), config(Vector4f::Zero()), feather(0.0f), opacity(0.0f) {};

    NGP_HOST_DEVICE static Mask3D Box(const Vector3f& dims, const Matrix4f& transform, const EMaskMode& mode, const float& feather, const float& opacity) {
        return Mask3D(EMaskShape::Box, transform, mode, Vector4f(dims.x(), dims.y(), dims.z(), 0.0f), feather, opacity);
    };

    NGP_HOST_DEVICE static Mask3D Cylinder(const float& radius, const float& height, const Matrix4f& transform, const EMaskMode& mode, const float& feather, const float& opacity) {
        return Mask3D(EMaskShape::Cylinder, transform, mode, Vector4f(radius, height, 0.0f, 0.0f), feather, opacity);
    };

    NGP_HOST_DEVICE static Mask3D Sphere(const float& radius, const Matrix4f& transform, const EMaskMode& mode, const float& feather, const float& opacity) {
        return Mask3D(EMaskShape::Sphere, transform, mode, Vector4f(radius, 0.0f, 0.0f, 0.0f), feather, opacity);
    };

    inline NGP_HOST_DEVICE float sample(const Vector3f& p) const {
        
        Matrix4f t_inv = transform.inverse();
        Vector4f p_h = p.homogeneous();
        Vector4f p_t = t_inv * p_h;
        Vector3f p_local = p_t.head<3>();

        float d = 0.0f;
        switch (shape) {
            case EMaskShape::Box:
                d = sdfBox(p_local, config.head<3>());
                break;
            case EMaskShape::Cylinder:
                d = sdfCylinder(p_local, config.x(), config.y());
                break;
            case EMaskShape::Sphere:
                d = sdfSphere(p_local, config.x());
                break;
        }

        // TODO: Decompose transform to get accurate feathering
        // need sdf from feather bounds also
        float alpha;
        if (feather == 0.0f) {
            alpha = d < 0.0f ? 1.0f : 0.0f;
        } else {
            alpha = tcnn::clamp(0.5f - d / feather, 0.0f, 1.0f);
        }
        return 2.0f * opacity * (alpha - 0.5f) * ((mode == EMaskMode::Add) ? 1.0f : -1.0f);
    };
};

NGP_NAMESPACE_END
