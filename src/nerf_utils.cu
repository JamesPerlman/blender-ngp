
#include <Eigen/Dense>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>

#include <tiny-cuda-nn/common.h>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

__host__ __device__ uint32_t get_grid_mip_offset(uint32_t mip, uint32_t grid_volume) {
	return grid_volume * mip;
}

__device__ float get_distance_to_next_voxel(const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) { // dda like step
	Vector3f p = res * pos;
	float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
	float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
	float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

__host__ __device__ float get_dt(float t, float cone_angle, float min_cone_stepsize, float max_cone_stepsize) {
	return tcnn::clamp(t * cone_angle, min_cone_stepsize, max_cone_stepsize);
}

__device__ float get_t_advanced_to_next_voxel(float t, float cone_angle, const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res, float min_cone_stepsize, float max_cone_stepsize) {
	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
	// due to the different stepping.
	// float dt = calc_dt(t, cone_angle);
	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + get_distance_to_next_voxel(pos, dir, idir, res);
	do {
		t += get_dt(t, cone_angle, min_cone_stepsize, max_cone_stepsize);
	} while (t < t_target);
	return t;
}

__device__ float get_network_rgb(float val, ENerfActivation activation) {
	switch (activation) {
	case ENerfActivation::None: return val;
	case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
	case ENerfActivation::Logistic: return tcnn::logistic(val);
	case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
	default: assert(false);
	}
	return 0.0f;
}

__device__ float get_network_rgb_derivative(float val, ENerfActivation activation) {
	switch (activation) {
	case ENerfActivation::None: return 1.0f;
	case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
	case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
	case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
	default: assert(false);
	}
	return 0.0f;
}

__device__ float get_network_density(float val, ENerfActivation activation) {
	switch (activation) {
	case ENerfActivation::None: return val;
	case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
	case ENerfActivation::Logistic: return tcnn::logistic(val);
	case ENerfActivation::Exponential: return __expf(val);
	default: assert(false);
	}
	return 0.0f;
}

__device__ float get_network_density_derivative(float val, ENerfActivation activation) {
	switch (activation) {
	case ENerfActivation::None: return 1.0f;
	case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
	case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
	case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -15.0f, 15.0f));
	default: assert(false);
	}
	return 0.0f;
}

__device__ Array3f get_network_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, ENerfActivation activation) {
	return {
		get_network_rgb(float(local_network_output[0]), activation),
		get_network_rgb(float(local_network_output[1]), activation),
		get_network_rgb(float(local_network_output[2]), activation)
	};
}

__device__ Vector3f get_warped_pos(const Vector3f& pos, const BoundingBox& aabb) {
	// return {tcnn::logistic(pos.x() - 0.5f), tcnn::logistic(pos.y() - 0.5f), tcnn::logistic(pos.z() - 0.5f)};
	// return pos;

	return aabb.relative_pos(pos);
}

__device__ Vector3f get_unwarped_pos(const Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.min + pos.cwiseProduct(aabb.diag());
}

__device__ Vector3f get_unwarped_pos_derivative(const Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.diag();
}

__device__ Vector3f get_warped_pos_derivative(const Vector3f& pos, const BoundingBox& aabb) {
	return get_unwarped_pos_derivative(pos, aabb).cwiseInverse();
}

__host__ __device__ Vector3f get_warped_dir(const Vector3f& dir) {
	return (dir + Vector3f::Ones()) * 0.5f;
}

__device__ Vector3f get_unwarped_dir(const Vector3f& dir) {
	return dir * 2.0f - Vector3f::Ones();
}

__device__ Vector3f get_warped_dir_derivative(const Vector3f& dir) {
	return Vector3f::Constant(0.5f);
}

__device__ Vector3f get_unwarped_dir_derivative(const Vector3f& dir) {
	return Vector3f::Constant(2.0f);
}

__device__ float get_warped_dt(float dt, float min_cone_stepsize, uint32_t nerf_cascades) {
	float max_stepsize = min_cone_stepsize * (1 << (nerf_cascades - 1));
	return (dt - min_cone_stepsize) / (max_stepsize - min_cone_stepsize);
}

__device__ float get_unwarped_dt(float dt, float min_cone_stepsize, uint32_t nerf_cascades) {
	float max_stepsize = min_cone_stepsize * (1 << (nerf_cascades - 1));
	return dt * (max_stepsize - min_cone_stepsize) + min_cone_stepsize;
}

__device__ int get_mip_from_pos(const Vector3f& pos, uint32_t max_cascade) {
	int exponent;
	float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
	frexpf(maxval, &exponent);
	return min(max_cascade, max(0, exponent + 1));
}

__device__ int get_mip_from_dt(float dt, const Vector3f& pos, uint32_t grid_size, uint32_t max_cascade) {
	int mip = get_mip_from_pos(pos, max_cascade);
	dt *= 2 * grid_size;
	if (dt < 1.f) return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(max_cascade, max(exponent, mip));
}

__device__ uint32_t get_cascaded_grid_idx_at(Vector3f pos, uint32_t mip, uint32_t grid_size) {
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= Vector3f::Constant(0.5f);
	pos *= mip_scale;
	pos += Vector3f::Constant(0.5f);

	Vector3i i = (pos * grid_size).cast<int>();

	if (i.x() < -1 || i.x() > grid_size || i.y() < -1 || i.y() > grid_size || i.z() < -1 || i.z() > grid_size) {
		printf("WTF %d %d %d\n", i.x(), i.y(), i.z());
	}

	uint32_t idx = tcnn::morton3D(
		tcnn::clamp(i.x(), 0, (int)grid_size - 1),
		tcnn::clamp(i.y(), 0, (int)grid_size - 1),
		tcnn::clamp(i.z(), 0, (int)grid_size - 1)
	);

	return idx;
}

__device__ bool get_is_density_grid_occupied_at(const Vector3f& pos, const uint8_t* density_grid_bitfield, uint32_t mip, uint32_t grid_size, uint32_t grid_volume) {
	uint32_t idx = get_cascaded_grid_idx_at(pos, mip, grid_size);
	return density_grid_bitfield[idx / 8 + get_grid_mip_offset(mip, grid_volume) / 8] & (1 << (idx % 8));
}

__device__ float get_cascaded_grid_at(Vector3f pos, const float* cascaded_grid, uint32_t mip, uint32_t grid_volume) {
	uint32_t idx = get_cascaded_grid_idx_at(pos, mip, grid_volume);
	return cascaded_grid[idx + get_grid_mip_offset(mip, grid_volume)];
}

__device__ float& get_cascaded_grid_at(Vector3f pos, float* cascaded_grid, uint32_t mip, uint32_t grid_volume) {
	uint32_t idx = get_cascaded_grid_idx_at(pos, mip, grid_volume);
	return cascaded_grid[idx + get_grid_mip_offset(mip, grid_volume)];
}

NGP_NAMESPACE_END
