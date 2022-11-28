/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   camera_models.h
 * 
 *  @author James Perlman, avid instant-ngp fan
 */

#pragma once

#include <Eigen/Dense>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/random_val.cuh>


NGP_NAMESPACE_BEGIN

enum class ECameraModel : int {
    Perspective,
    QuadrilateralHexahedron,
    SphericalQuadrilateral,
};

struct Quadrilateral3D {
    Eigen::Vector3f tl;
    Eigen::Vector3f tr;
    Eigen::Vector3f bl;
    Eigen::Vector3f br;
    
	NGP_HOST_DEVICE Quadrilateral3D(const Eigen::Vector3f& tl, const Eigen::Vector3f& tr, const Eigen::Vector3f& bl, const Eigen::Vector3f& br) : tl(tl), tr(tr), bl(bl), br(br) {};
	
	NGP_HOST_DEVICE static Quadrilateral3D Zero() {
		return Quadrilateral3D(Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
	};

    Eigen::Vector3f center() const {
        return (tl + tr + bl + br) / 4.0f;
    };

	bool operator==(const Quadrilateral3D& other) const {
		return tl == other.tl && tr == other.tr && bl == other.bl && br == other.br;
	};

	bool operator!=(const Quadrilateral3D& other) const {
		return !(*this == other);
	};
};

struct QuadrilateralHexahedron {
    Quadrilateral3D front;
    Quadrilateral3D back;

    NGP_HOST_DEVICE QuadrilateralHexahedron(const Quadrilateral3D& front, const Quadrilateral3D& back) : front(front), back(back) {};
    
	NGP_HOST_DEVICE static QuadrilateralHexahedron Zero() {
		return QuadrilateralHexahedron(Quadrilateral3D::Zero(), Quadrilateral3D::Zero());
	};

    Eigen::Vector3f center() const {
        return (front.center() + back.center()) / 2.0f;
    };

	bool operator==(const QuadrilateralHexahedron& other) const {
		return front == other.front && back == other.back;
	};
	bool operator!=(const QuadrilateralHexahedron& other) const {
		return !(*this == other);
	};
};

inline NGP_HOST_DEVICE Ray quadrilateral_hexahedron_pixel_to_ray(
    const uint32_t spp,
    const Eigen::Vector2i& pixel,
    const Eigen::Vector2i& resolution,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
    const QuadrilateralHexahedron& qh,
    float near_distance = 0.0f,
    float focus_z = 0.0f,
    float aperture_size = 0.0f
) {
    Eigen::Vector2f uv = (pixel.cast<float>() + Eigen::Vector2f(0.5f, 0.5f)).array() / resolution.cast<float>().array();
    
    Eigen::Vector3f front_ab = qh.front.tl + uv.x() * (qh.front.tr - qh.front.tl);
    Eigen::Vector3f front_dc = qh.front.bl + uv.x() * (qh.front.br - qh.front.bl);
    Eigen::Vector3f front_p = front_ab + uv.y() * (front_dc - front_ab);

    Eigen::Vector3f back_ab = qh.back.tl + uv.x() * (qh.back.tr - qh.back.tl);
    Eigen::Vector3f back_dc = qh.back.bl + uv.x() * (qh.back.br - qh.back.bl);
    Eigen::Vector3f back_p = back_ab + uv.y() * (back_dc - back_ab);

    Eigen::Vector3f dir = front_p - back_p;
	dir /= dir.z();
    Eigen::Vector3f origin = back_p;

	origin = camera_matrix.block<3, 3>(0, 0) * origin + camera_matrix.col(3);
	dir = camera_matrix.block<3, 3>(0, 0) * dir;

    if (aperture_size > 0.0f) {
		Eigen::Vector3f lookat = origin + dir * focus_z;
		Eigen::Vector2f blur = aperture_size * square2disk_shirley(ld_random_val_2d(spp, (uint32_t)pixel.x() * 19349663 + (uint32_t)pixel.y() * 96925573) * 2.0f - Eigen::Vector2f::Ones());
		origin += camera_matrix.block<3, 2>(0, 0) * blur;
		dir = (lookat - origin) / focus_z;
	}

    origin = origin + dir * near_distance;

    return {origin, dir};
};

struct SphericalQuadrilateral {
	float width;
	float height;
	float curvature;

	NGP_HOST_DEVICE SphericalQuadrilateral(const float& width, const float& height, const float& curvature) : width(width), height(height), curvature(curvature) {};
	NGP_HOST_DEVICE static SphericalQuadrilateral Zero() {
		return SphericalQuadrilateral(0.0f, 0.0f, 0.0f);
	};

	bool operator==(const SphericalQuadrilateral& other) const {
		return width == other.width && height == other.height && curvature == other.curvature;
	};
	bool operator!=(const SphericalQuadrilateral& other) const {
		return !(*this == other);
	};
};

inline NGP_HOST_DEVICE Eigen::Vector2f walk_along_circle(float curvature, float linear_len, float arc_len)
{
	float arc_t = arc_len / (2.0f * linear_len);

	if (arc_t == 0.0f || linear_len == 0.0f) {
		return Eigen::Vector2f(0.0f, 0.0f);
	}

	if (curvature == 0.0f) {
		return Eigen::Vector2f(linear_len * arc_t, 0.0f);
	}

	float tpc = 2.0 * PI() * curvature;
	float s_tpc = linear_len / tpc;
	return s_tpc * Eigen::Vector2f(sinf(tpc * arc_t), 1.0f - cosf(tpc * arc_t));
};

inline NGP_HOST_DEVICE Eigen::Vector3f walk_along_sphere(float curvature, float max_linear_len, float azimuth, float arc_len)
{
	Eigen::Vector2f rz = walk_along_circle(curvature, max_linear_len, arc_len);
	return Eigen::Vector3f(rz.x() * cosf(azimuth), rz.x() * sinf(azimuth), rz.y());
};

inline NGP_HOST_DEVICE Ray spherical_quadrilateral_pixel_to_ray(
    const uint32_t spp,
    const Eigen::Vector2i& pixel,
    const Eigen::Vector2i& resolution,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
    const SphericalQuadrilateral& sq,
    float near_distance = 0.0f,
    float focus_z = 0.0f,
    float aperture_size = 0.0f
) {
	Eigen::Vector2f sq_dims = Eigen::Vector2f(sq.width, sq.height);
	float max_linear_len = sq_dims.norm();

	// Normalize from [-0.5, -0.5] to [0.5, 0.5]
	Eigen::Vector2f uv = 2.0 * ((pixel.cast<float>() + Eigen::Vector2f(0.5f, 0.5f)).array() / resolution.cast<float>().array() - Eigen::Array2f(0.5f, 0.5f));
	Eigen::Vector2f xy = Eigen::Vector2f(sq_dims.x() * uv.x(), sq_dims.y() * uv.y());
	float a = atan2f(xy.y(), xy.x());
	float r = xy.norm();

	Eigen::Vector3f origin = walk_along_sphere(sq.curvature, max_linear_len, a, r);
	Eigen::Vector3f dir = Eigen::Vector3f(0.0f, 0.0f, 1.0f);

	if (sq.curvature != 0.0f) {
		// calculate sphere center
		Eigen::Vector3f sc = Eigen::Vector3f(0.0f, 0.0f, max_linear_len / (2.0f * PI() * sq.curvature));
		float k = sq.curvature > 0.0f ? 1.0f : -1.0f;
		dir = k * (sc - origin).normalized();
	}

	origin = camera_matrix.block<3, 3>(0, 0) * origin + camera_matrix.col(3);
	dir = camera_matrix.block<3, 3>(0, 0) * dir;

	if (aperture_size > 0.0f) {
		Eigen::Vector3f lookat = origin + dir * focus_z;
		Eigen::Vector2f blur = aperture_size * square2disk_shirley(ld_random_val_2d(spp, (uint32_t)pixel.x() * 19349663 + (uint32_t)pixel.y() * 96925573) * 2.0f - Eigen::Vector2f::Ones());
		origin += camera_matrix.block<3, 2>(0, 0) * blur;
		dir = (lookat - origin) / focus_z;
	}

	origin = origin + dir * near_distance;

	return {origin, dir};
};


inline NGP_HOST_DEVICE Ray perspective_pixel_to_ray(
	uint32_t spp,
	const Eigen::Vector2i& pixel,
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	float near_distance = 0.0f,
	float aperture_size = 0.0f,
	float focus_z = 1.0f
) {
	Eigen::Vector2f offset = ld_random_pixel_offset(spp);
	Eigen::Vector2f uv = (pixel.cast<float>() + offset).cwiseQuotient(resolution.cast<float>());

	Eigen::Vector3f dir;

	dir = {
		(uv.x() - 0.5f) * (float)resolution.x() / focal_length.x(),
		(uv.y() - 0.5f) * (float)resolution.y() / focal_length.y(),
		1.0f
	};

	Eigen::Vector3f head_pos = {0.0f, 0.0f, 0.f};
	dir = camera_matrix.block<3, 3>(0, 0) * dir;

	Eigen::Vector3f origin = camera_matrix.block<3, 3>(0, 0) * head_pos + camera_matrix.col(3);
	
	if (aperture_size > 0.0f) {
		Eigen::Vector3f lookat = origin + dir * focus_z;
		Eigen::Vector2f blur = aperture_size * square2disk_shirley(ld_random_val_2d(spp, (uint32_t)pixel.x() * 19349663 + (uint32_t)pixel.y() * 96925573) * 2.0f - Eigen::Vector2f::Ones());
		origin += camera_matrix.block<3, 2>(0, 0) * blur;
		dir = (lookat - origin) / focus_z;
	}
	
	origin += dir * near_distance;

	return {origin, dir};
};


NGP_NAMESPACE_END
