/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_scene.cuh
 * 
 *  @author James Perlman, avid instant-ngp fan
 */

#pragma once

#include <filesystem/path.h>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <fmt/core.h>
#include <json/json.hpp>
#include <tiny-cuda-nn/common.h>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/random_val.cuh>
#include <neural-graphics-primitives/render_modifiers.cuh>

using namespace Eigen;
using namespace tcnn;
using json = nlohmann::json;
namespace fs = ::filesystem;
NGP_NAMESPACE_BEGIN




static const size_t SNAPSHOT_FORMAT_VERSION = 1;

struct NerfScene {
    fs::path snapshot_path;
    BoundingBox aabb = BoundingBox{Vector3f::Constant(0.5f), Vector3f::Constant(0.5f)};
    RenderModifiers modifiers;
    Eigen::Matrix4f transform;

    // these should be inaccessible to python or the initializer
    tcnn::GPUMemory<float> density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
    tcnn::GPUMemory<uint8_t> density_grid_bitfield;
    tcnn::GPUMemory<float> density_grid_mean;
    
    ENerfActivation rgb_activation = ENerfActivation::Logistic;
    ENerfActivation density_activation = ENerfActivation::Exponential;

    uint32_t grid_size = 128;
    uint32_t max_cascade = 0;
    uint32_t num_cascades = 8;
    float min_optical_thickness = 0.01f;
    float min_transmittance = 0.01f;
    float cone_angle_constant = 1.0f / 256.0f; // TODO: if aabb_scale <= 1 this is 0.0f

    nlohmann::json network_config = {};

    struct Inference {

        struct LevelStats {
            float mean() { return count ? (x / (float)count) : 0.f; }
            float variance() { return count ? (xsquared - (x * x) / (float)count) / (float)count : 0.f; }
            float sigma() { return sqrtf(variance()); }
            float fraczero() { return (float)numzero / float(count + numzero); }
            float fracquant() { return (float)numquant / float(count); }

            float x;
            float xsquared;
            float min;
            float max;
            int numzero;
            int numquant;
            int count;
        };
        
        default_rng_t rng;
        uint32_t aabb_scale = 1;
        uint32_t num_levels = 0;
        uint32_t base_grid_resolution;
        float desired_resolution = 2048.0f; // Desired resolution of the finest hashgrid level over the unit cube
        float per_level_scale;
        NetworkDims network_dims;
        std::shared_ptr<NerfNetwork<precision_t>> network;
        std::shared_ptr<tcnn::Encoding<precision_t>> encoding;
        uint32_t n_encoding_params;
        uint32_t n_extra_dims = 0;

        std::vector<LevelStats> level_stats;
        std::vector<LevelStats> first_layer_column_stats;

        
		RaysNerfSoa rays[2];
		RaysNerfSoa rays_hit;
		precision_t* network_output;
		float* network_input;
		tcnn::GPUMemory<uint32_t> hit_counter;
		tcnn::GPUMemory<uint32_t> alive_counter;
		uint32_t n_rays_initialized = 0;
        uint32_t n_rays_alive = 0;
		tcnn::GPUMemoryArena::Allocation scratch_alloc = {};

        uint32_t min_steps_inbetween_compaction = 1;
        uint32_t max_steps_inbetween_compaction = 8;

        Inference(json config = {}) {
            if (config.empty()) {
                return;
            }

            rng = default_rng_t{1337};

            json& encoding_config = config["encoding"];
            json& loss_config = config["loss"];
            json& optimizer_config = config["optimizer"];
            json& network_config = config["network"];

            network_dims = {};
            network_dims.n_input = sizeof(NerfCoordinate) / sizeof(float);
            network_dims.n_output = 4;
            network_dims.n_pos = sizeof(NerfPosition) / sizeof(float);

            // Some of the Nerf-supported losses are not supported by tcnn::Loss,
            // so just create a dummy L2 loss there. The NeRF code path will bypass
            // the tcnn::Loss in any case.
            loss_config["otype"] = "L2";

            aabb_scale = config["snapshot"]["nerf"]["aabb_scale"];

            // Automatically determine certain parameters if we're dealing with the (hash)grid encoding
            if (to_lower(encoding_config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
                encoding_config["n_pos_dims"] = network_dims.n_pos;

                const uint32_t n_features_per_level = encoding_config.value("n_features_per_level", 2u);

                if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
                    num_levels = (uint32_t)encoding_config["n_features"] / n_features_per_level;
                } else {
                    num_levels = encoding_config.value("n_levels", 16u);
                }

                level_stats.resize(num_levels);
                first_layer_column_stats.resize(num_levels);

                const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);

                base_grid_resolution = encoding_config.value("base_resolution", 0);
                if (!base_grid_resolution) {
                    base_grid_resolution = 1u << ((log2_hashmap_size) / network_dims.n_pos);
                    encoding_config["base_resolution"] = base_grid_resolution;
                }

               

                // Automatically determine suitable per_level_scale
                per_level_scale = encoding_config.value("per_level_scale", 0.0f);
                if (per_level_scale <= 0.0f && num_levels > 1) {
                    per_level_scale = std::exp(std::log(desired_resolution * (float)aabb_scale / (float)base_grid_resolution) / (num_levels - 1));
                    encoding_config["per_level_scale"] = per_level_scale;
                }

                tlog::info()
                    << "GridEncoding: "
                    << " Nmin=" << base_grid_resolution
                    << " b=" << per_level_scale
                    << " F=" << n_features_per_level
                    << " T=2^" << log2_hashmap_size
                    << " L=" << num_levels
                    ;
            }


            json& dir_encoding_config = config["dir_encoding"];
            json& rgb_network_config = config["rgb_network"];

            uint32_t n_dir_dims = 3;
            network = std::make_shared<NerfNetwork<precision_t>>(
                network_dims.n_pos,
                n_dir_dims,
                n_extra_dims,
                network_dims.n_pos + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
                encoding_config,
                dir_encoding_config,
                network_config,
                rgb_network_config
            );

            encoding = network->encoding();
            n_encoding_params = encoding->n_params() + network->dir_encoding()->n_params();

            tlog::info()
                << "Density model: " << network_dims.n_pos
                << "--[" << std::string(encoding_config["otype"])
                << "]-->" << network->encoding()->padded_output_width()
                << "--[" << std::string(network_config["otype"])
                << "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"]+2) << ")"
                << "]-->" << 1
                ;

            tlog::info()
                << "Color model:   " << n_dir_dims
                << "--[" << std::string(dir_encoding_config["otype"])
                << "]-->" << network->dir_encoding()->padded_output_width() << "+" << network_config.value("n_output_dims", 16u)
                << "--[" << std::string(rgb_network_config["otype"])
                << "(neurons=" << (int)rgb_network_config["n_neurons"] << ",layers=" << ((int)rgb_network_config["n_hidden_layers"]+2) << ")"
                << "]-->" << 3
                ;

            size_t n_network_params = network->n_params() - n_encoding_params;

            tlog::info() << "  total_encoding_params=" << n_encoding_params << " total_network_params=" << n_network_params;
        };

        void enlarge_workspace(size_t n_elements, cudaStream_t stream) {
            n_elements = next_multiple(n_elements, size_t(tcnn::batch_size_granularity));
            size_t num_floats = sizeof(NerfCoordinate) / 4 + n_extra_dims;
            size_t padded_output_width = network->padded_output_width();
            auto scratch = allocate_workspace_and_distribute<
                Array4f, float, NerfPayload, // m_rays[0]
                Array4f, float, NerfPayload, // m_rays[1]
                Array4f, float, NerfPayload, // m_rays_hit
                network_precision_t,
                float
            >(
                stream, &scratch_alloc,
                n_elements, n_elements, n_elements,
                n_elements, n_elements, n_elements,
                n_elements, n_elements, n_elements,
                n_elements * max_steps_inbetween_compaction * padded_output_width,
                n_elements * max_steps_inbetween_compaction * num_floats
            );

            rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
            rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
            rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

            network_output = std::get<9>(scratch);
            network_input = std::get<10>(scratch);
        }

        void clear_workspace() {
            scratch_alloc = {};
        }
    } inference = {};

    // TODO:
    // const NerfTrainingConfig training_config;

    inline NGP_HOST_DEVICE uint32_t grid_volume() const { return grid_size * grid_size * grid_size; };

    json load_snapshot_config(const fs::path& path) {
        tlog::info() << "Loading network config from: " << path;

        if (path.empty() || !path.exists()) {
            throw std::runtime_error{fmt::format("Network config {} does not exist.", path.str())};
        }

        std::ifstream f{path.str(), std::ios::in | std::ios::binary};
        return json::from_msgpack(f);
    }

    inline NGP_HOST_DEVICE uint32_t grid_mip_offset(uint32_t mip) const {
        return grid_volume() * mip;
    }

    inline NGP_HOST_DEVICE uint8_t* get_density_grid_bitfield_mip(uint32_t mip) {
	    return density_grid_bitfield.data() + grid_mip_offset(mip)/8;
    }
    
    void update_density_grid_mean_and_bitfield(cudaStream_t stream);

    void load_snapshot() {
        if (snapshot_path.empty()) {
            throw std::runtime_error{"No snapshot path specified."};
        }

        auto config = load_snapshot_config(snapshot_path);
        if (!config.contains("snapshot")) {
            throw std::runtime_error{fmt::format("File {} does not contain a snapshot.", snapshot_path.str())};
        }

        const auto& snapshot = config["snapshot"];

        // TODO: most of this should go into Inference and Inference should be renamed to SceneData

        if (snapshot.value("version", 0) < SNAPSHOT_FORMAT_VERSION) {
            throw std::runtime_error{"Snapshot uses an old format."};
        }
        
        grid_size = snapshot["density_grid_size"];
        aabb = snapshot.value("aabb", aabb);

        GPUMemory<__half> density_grid_fp16 = snapshot["density_grid_binary"];
        density_grid.resize(density_grid_fp16.size());

        parallel_for_gpu(density_grid_fp16.size(), [density_grid=density_grid.data(), density_grid_fp16=density_grid_fp16.data()] __device__ (size_t i) {
            density_grid[i] = (float)density_grid_fp16[i];
        });

        if (density_grid.size() == grid_volume() * (max_cascade + 1)) {
            update_density_grid_mean_and_bitfield(nullptr);
        } else if (density_grid.size() != 0) {
            // A size of 0 indicates that the density grid was never populated, which is a valid state of a (yet) untrained model.
            throw std::runtime_error{"Incompatible number of grid cascades."};
        }

        network_config = config;

        inference = Inference(config);
    }

    NerfScene(const NerfScene& other) {
        snapshot_path = other.snapshot_path;
        aabb = other.aabb;
        modifiers = other.modifiers;
        transform = other.transform;
    }

    NerfScene operator=(const NerfScene& other) {
        if (this != &other) {
            snapshot_path = other.snapshot_path;
            aabb = other.aabb;
            modifiers = other.modifiers;
            transform = other.transform;
        }

        return *this;
    }
    
};

NGP_NAMESPACE_END
