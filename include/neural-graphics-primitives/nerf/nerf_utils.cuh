
#include <Eigen/Dense>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/common.h>

NGP_NAMESPACE_BEGIN

__host__ __device__ uint32_t get_grid_mip_offset(uint32_t mip, uint32_t grid_volume);

__device__ float get_distance_to_next_voxel(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir, const Eigen::Vector3f& idir, uint32_t res);

__host__ __device__ float get_dt(float t, float cone_angle, float min_cone_stepsize, float max_cone_stepsize);

__device__ float get_t_advanced_to_next_voxel(float t, float cone_angle, const Eigen::Vector3f& pos, const Eigen::Vector3f& dir, const Eigen::Vector3f& idir, uint32_t res, float min_cone_stepsize, float max_cone_stepsize);

__device__ float get_network_rgb(float val, ENerfActivation activation);

__device__ float get_network_rgb_derivative(float val, ENerfActivation activation);

__device__ float get_network_density(float val, ENerfActivation activation);

__device__ float get_network_density_derivative(float val, ENerfActivation activation);

__device__ Eigen::Array3f get_network_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, ENerfActivation activation);

__device__ Eigen::Vector3f get_warped_pos(const Eigen::Vector3f& pos, const BoundingBox& aabb);

__device__ Eigen::Vector3f get_unwarped_pos(const Eigen::Vector3f& pos, const BoundingBox& aabb);

__device__ Eigen::Vector3f get_unwarped_pos_derivative(const Eigen::Vector3f& pos, const BoundingBox& aabb);

__device__ Eigen::Vector3f get_warped_pos_derivative(const Eigen::Vector3f& pos, const BoundingBox& aabb);

__host__ __device__ Eigen::Vector3f get_warped_dir(const Eigen::Vector3f& dir);

__device__ Eigen::Vector3f get_unwarped_dir(const Eigen::Vector3f& dir);

__device__ Eigen::Vector3f get_warped_dir_derivative(const Eigen::Vector3f& dir);

__device__ Eigen::Vector3f get_unwarped_dir_derivative(const Eigen::Vector3f& dir);

__device__ int get_mip_from_pos(const Eigen::Vector3f& pos, uint32_t max_cascade);

__device__ int get_mip_from_dt(float dt, const Eigen::Vector3f& pos, uint32_t grid_size, uint32_t max_cascade);

__device__ float get_warped_dt(float dt, float min_cone_stepsize, uint32_t nerf_cascades);

__device__ float get_unwarped_dt(float dt, float min_cone_stepsize, uint32_t nerf_cascades);

__device__ uint32_t get_cascaded_grid_idx_at(Eigen::Vector3f pos, uint32_t mip, uint32_t grid_size);

__device__ bool get_is_density_grid_occupied_at(const Eigen::Vector3f& pos, const uint8_t* density_grid_bitfield, uint32_t mip, uint32_t grid_size, uint32_t grid_volume);

__device__ float get_cascaded_grid_at(Eigen::Vector3f pos, const float* cascaded_grid, uint32_t mip, uint32_t grid_volume);

__device__ float& get_cascaded_grid_at(Eigen::Vector3f pos, float* cascaded_grid, uint32_t mip, uint32_t grid_volume);

NGP_NAMESPACE_END
