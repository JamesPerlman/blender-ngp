/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   mask_shapes.h
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
    All,
};

// thank you https://iquilezles.org/articles/distfunctions/ for the sdf primitives

// xyz -> yzx

inline NGP_HOST_DEVICE float sdf_box(const Eigen::Vector3f& p, const Eigen::Vector3f& b) {
    Eigen::Vector3f d = p.cwiseAbs() - 0.5f * b;
    return d.cwiseMax(0.0f).norm() + min(max(d.x(), max(d.y(), d.z())), 0.0f);
};

inline NGP_HOST_DEVICE float sdf_cylinder(const Eigen::Vector3f& p, const float& r, const float& h) {
    Eigen::Vector2f d = Eigen::Vector2f(Eigen::Vector2f(p.y(), p.x()).norm(), p.z()).cwiseAbs() - Eigen::Vector2f(r, 0.5f * h);
    return d.cwiseMax(0.0f).norm() + min(max(d.x(), d.y()), 0.0f);
};

inline NGP_HOST_DEVICE float sdf_sphere(const Eigen::Vector3f& p, const float& r) {
    return p.norm() - r;
};

// intersections


inline NGP_HOST_DEVICE bool ray_intersects_box(const Ray& ray, const Eigen::Vector3f& box_size) {
    const Eigen::Vector3f inv_dir = ray.d.cwiseInverse();
    const Eigen::Vector3f t0 = -0.5f * box_size - ray.o;
    const Eigen::Vector3f t1 = 0.5f * box_size - ray.o;
    const Eigen::Vector3f t0_cpi = t0.cwiseProduct(inv_dir);
    const Eigen::Vector3f t1_cpi = t1.cwiseProduct(inv_dir);
    const Eigen::Vector3f tmin = t0_cpi.cwiseMin(t1_cpi);
    const Eigen::Vector3f tmax = t0_cpi.cwiseMax(t1_cpi);
    return tmin.maxCoeff() <= tmax.minCoeff();
};


inline NGP_HOST_DEVICE bool ray_intersects_sphere(const Ray& ray, const float& radius) {
    const float a = powf(ray.d.dot(ray.o), 2.0);
    const float b = ray.o.squaredNorm() - radius * radius;
    return !((a - b) < 0.0f);
};

inline NGP_HOST_DEVICE bool ray_intersects_plane(const Ray& ray, const Eigen::Vector3f& n, const Eigen::Vector3f& p) {
    const float denom = n.dot(ray.d);
    if (denom > 1e-6f) {
        return (p - ray.o).dot(n) / denom >= 0.0f;
    }
    return false;
};

inline NGP_HOST_DEVICE bool intersect_plane_ray(const Ray& ray, const Eigen::Vector3f& n, const Eigen::Vector3f& p, float& t) {
    const float denom = n.dot(ray.d);
    if (denom > 1e-6f) {
        t = (p - ray.o).dot(n) / denom;
        return t >= 0.0f;
    }
    return false;
};

inline NGP_HOST_DEVICE bool ray_intersects_cylinder(const Ray& ray, const float& radius, const float& height) {
    const float a = ray.d.head<2>().squaredNorm();
    const float b = 2.0f * ray.d.head<2>().dot(ray.o.head<2>());
    const float c = ray.o.head<2>().squaredNorm() - radius * radius;
    const float d = b * b - 4.0f * a * c;
    if (d < 0.0f) {
        return false;
    }

    const float d_sqrt = sqrtf(d);
    const float a2 = 2.0f * a;
    float h_2 = 0.5f * height;

    if (a2 > 1e-6f) {
        const float t0 = (-b - d_sqrt) / a2;
        const float t1 = (-b + d_sqrt) / a2;
        const float z0 = ray.o.z() + t0 * ray.d.z();
        const float z1 = ray.o.z() + t1 * ray.d.z();
        if ((z0 >= -h_2 && z0 <= h_2) || (z1 >= -h_2 && z1 <= h_2)) {
            return true;
        };
    }

    // calculate point where ray intersects cylinder endcaps
    float t = 0.0f;
    if (intersect_plane_ray(ray, Eigen::Vector3f(0.0f, 0.0f, 1.0f), Eigen::Vector3f(0.0f, 0.0f, h_2), t)) {
        const Eigen::Vector3f p = ray.o + t * ray.d;
        if (p.head<2>().squaredNorm() <= radius * radius) {
            return true;
        }
    }
    
    if (intersect_plane_ray(ray, Eigen::Vector3f(0.0f, 0.0f, -1.0f), Eigen::Vector3f(0.0f, 0.0f, -h_2), t)) {
        const Eigen::Vector3f p = ray.o + t * ray.d;
        if (p.head<2>().squaredNorm() <= radius * radius) {
            return true;
        }
    }

    return false;
};

#define ConfigArray Eigen::Array<float, 6, 1>

// TODO: split into descriptor and functional gpu instance (maybe?)
struct Mask3D {
    EMaskMode mode;
    EMaskShape shape;
    Eigen::Matrix4f transform;
    ConfigArray config;
    float feather;
    float opacity;

    NGP_HOST_DEVICE Mask3D(const EMaskShape& shape, const Eigen::Matrix4f& transform, const EMaskMode& mode, const ConfigArray& config, const float& feather, const float& opacity)
        : mode(mode), shape(shape), transform(transform), config(config), feather(feather), opacity(opacity) {};
    
    NGP_HOST_DEVICE Mask3D() : mode(EMaskMode::Add), shape(EMaskShape::Box), transform(Eigen::Matrix4f::Identity()), config({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}), feather(0.0f), opacity(0.0f) {};

    NGP_HOST_DEVICE static Mask3D All(const EMaskMode& mode) {
        return Mask3D(EMaskShape::All, Eigen::Matrix4f::Identity(), mode, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 0.0f, 1.0f);
    };

    NGP_HOST_DEVICE static Mask3D Box(const Eigen::Vector3f& dims, const Eigen::Matrix4f& transform, const EMaskMode& mode, const float& feather, const float& opacity) {
        return Mask3D(EMaskShape::Box, transform, mode, {dims.x(), dims.y(), dims.z(), 0.0f, 0.0f, 0.0f}, feather, opacity);
    };

    NGP_HOST_DEVICE static Mask3D Cylinder(const float& radius, const float& height, const Eigen::Matrix4f& transform, const EMaskMode& mode, const float& feather, const float& opacity) {
        return Mask3D(EMaskShape::Cylinder, transform, mode, {radius, height, 0.0f, 0.0f, 0.0f, 0.0f}, feather, opacity);
    };

    NGP_HOST_DEVICE static Mask3D Sphere(const float& radius, const Eigen::Matrix4f& transform, const EMaskMode& mode, const float& feather, const float& opacity) {
        return Mask3D(EMaskShape::Sphere, transform, mode, {radius, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, feather, opacity);
    };

    inline NGP_HOST_DEVICE float sample(const Eigen::Vector3f& p) const {

        Eigen::Matrix4f t_inv = transform.inverse();
        Eigen::Vector4f p_h = p.homogeneous();
        Eigen::Vector4f p_t = t_inv * p_h;
        Eigen::Vector3f p_local = p_t.head<3>();

        float d = 0.0f;
        switch (shape) {
            case EMaskShape::Box:
                d = sdf_box(p_local, config.head<3>());
                break;
            case EMaskShape::Cylinder:
                d = sdf_cylinder(p_local, config.coeff(0), config.coeff(1));
                break;
            case EMaskShape::Sphere:
                d = sdf_sphere(p_local, config.coeff(0));
                break;
            case EMaskShape::All:
                return (mode == EMaskMode::Add) ? 1.0f : -1.0f;
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
        return opacity * alpha * ((mode == EMaskMode::Add) ? 1.0f : -1.0f);
    };

    inline NGP_HOST_DEVICE bool intersects_ray(const Ray& ray) const {
        // all subtract masks have infinite additive area outside of them, and they are all finite shapes
        if (mode == EMaskMode::Subtract) {
            return true;
        }
        
        Eigen::Matrix4f t_inv4x4 = transform.inverse();
        Eigen::Matrix3f t_inv3x3 = t_inv4x4.topLeftCorner<3, 3>();
        Eigen::Vector3f ray_o_local = (t_inv4x4 * ray.o.homogeneous()).head<3>();
        Eigen::Vector3f ray_d_local = t_inv3x3 * ray.d;
        Ray ray_local = {ray_o_local, ray_d_local.normalized()};

        switch (shape) {
            case EMaskShape::Box: {
                const auto box_dims = config.head<3>();
                return ray_intersects_box(ray_local, box_dims + 0.5f * feather);
            }
            case EMaskShape::Cylinder: {
                const float radius = config.coeff(0);
                const float height = config.coeff(1);
                return ray_intersects_cylinder(ray_local, radius + 0.5f * feather, height + 0.5f * feather);
            }
            case EMaskShape::Sphere: {
                const float radius = config.coeff(0);
                return ray_intersects_sphere(ray_local, radius + 0.5f * feather);
            }
            case EMaskShape::All:
                return false;
        }
    };
};

NGP_NAMESPACE_END
