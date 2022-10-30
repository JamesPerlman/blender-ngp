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

inline NGP_HOST_DEVICE float sdfBox(const Eigen::Vector3f& p) {
    Eigen::Vector3f d = p.cwiseAbs() - Eigen::Vector3f(1.0f, 1.0f, 1.0f);
    return d.cwiseMax(0.0f).norm() + std::min(std::max(d.x(), std::max(d.y(), d.z())), 0.0f);
};

inline NGP_HOST_DEVICE float sdfCylinder(const Eigen::Vector3f& p) {
    Eigen::Vector2f d = Eigen::Vector2f(Eigen::Vector2f(p.x(), p.z()).norm(), p.y()).cwiseAbs() - Eigen::Vector2f(1.0f, 0.5f);
    return d.cwiseMax(0.0f).norm() + std::min(std::max(d.x(), d.y()), 0.0f);
};

inline NGP_HOST_DEVICE float sdfSphere(const Eigen::Vector3f& p) {
    return p.norm() - 1.0f;
};

struct Mask3D {
    EMaskMode mode;
    EMaskShape shape;
    Eigen::Matrix4f transform;
    float feather;

    NGP_HOST_DEVICE Mask3D(const EMaskMode& mode, const EMaskShape& shape, const Eigen::Matrix4f& transform, const float& feather)
        : mode(mode), shape(shape), transform(transform), feather(feather) {};
    
    NGP_HOST_DEVICE Mask3D() : mode(EMaskMode::Add), shape(EMaskShape::Box), transform(Eigen::Matrix4f::Identity()), feather(0.0f) {};

    inline NGP_HOST_DEVICE float get_alpha(const Eigen::Vector3f& p) const {
        Eigen::Vector3f p_local = (transform.inverse() * p.homogeneous()).head<3>();
        float d = 0.0f;
        switch (shape) {
            case EMaskShape::Box:
                d = sdfBox(p_local);
                break;
            case EMaskShape::Cylinder:
                d = sdfCylinder(p_local);
                break;
            case EMaskShape::Sphere:
                d = sdfSphere(p_local);
                break;
        }

        float alpha;
        if (feather == 0.0f) {
            alpha = d < 0.0f ? 1.0f : 0.0f;
        } else {
            alpha = tcnn::clamp(0.5f + d / feather, 0.0f, 1.0f);
        }
        return 2.0f * alpha * ((mode == EMaskMode::Add) ? 1.0f : -1.0f);
    };
};

NGP_NAMESPACE_END
