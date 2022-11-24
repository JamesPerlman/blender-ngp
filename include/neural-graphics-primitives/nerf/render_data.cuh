/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_data.cuh
 * 
 *  @author James Perlman, avid instant-ngp fan
 */

#pragma once

#include <Eigen/Dense>

#include <neural-graphics-primitives/camera_models.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/nerf_render_proxy.cuh>
#include <neural-graphics-primitives/nerf/nerf_props.cuh>
#include <neural-graphics-primitives/nerf/neural_radiance_field.cuh>
#include <neural-graphics-primitives/nerf/render_data_workspace.cuh>
#include <neural-graphics-primitives/nerf/render_request.cuh>
#include <neural-graphics-primitives/nerf/render_modifiers.cuh>

#include <tiny-cuda-nn/common.h>

NGP_NAMESPACE_BEGIN

struct RenderData {
private:

	std::vector<NeuralRadianceField> _nerfs;
	std::vector<NerfRenderProxy> _proxies;

public:

    RenderOutputProperties output;
    RenderCameraProperties camera;
	RenderDataWorkspace workspace;
	tcnn::GPUMemory<NerfProps> nerf_props;

    RenderData() {};

    void update(const RenderRequest& request) {
        output = request.output;
        camera = request.camera;
        update_nerfs(request.nerfs, request.modifiers);
    }

    void update_nerfs(const std::vector<NerfDescriptor>& descriptors, const RenderModifiersDescriptor& global_modifiers) {
        // delete if nerf has no matching descriptor
        static_cast<void>( // discard result from remove_if iterator
            std::remove_if(_nerfs.begin(), _nerfs.end(), [&](const NeuralRadianceField& nerf) {
                auto result = std::find_if(descriptors.begin(), descriptors.end(), [&](const NerfDescriptor& descriptor) {
                    return descriptor.snapshot_path == nerf.snapshot_path;
                });
                // if result was not found, remove this nerf
                return result == descriptors.end();
            })
        );

        // add new nerfs
        for (const NerfDescriptor& desc : descriptors) {
            auto result = std::find_if(_nerfs.begin(), _nerfs.end(), [&](const NeuralRadianceField& nerf) {
                return nerf.snapshot_path == desc.snapshot_path;
            });
            if (result == _nerfs.end()) {
                _nerfs.emplace_back(desc);
            }
        }

		// update render proxies
		_proxies.clear();


		for (const NerfDescriptor& desc : descriptors) {
			auto nerf = std::find_if(_nerfs.begin(), _nerfs.end(), [&](const NeuralRadianceField& nerf) {
				return nerf.snapshot_path == desc.snapshot_path;
			});

			if (nerf == _nerfs.end()) {
				printf("Somehow a nerf was not found for this descriptor.  Something is very, very wrong.\n");
				continue;
			}

			_proxies.emplace_back(desc, *nerf, global_modifiers);
		}
    }

	void copy_from_host() {
		std::vector<NerfProps> _nerf_props;
		_nerf_props.reserve(_nerfs.size());
		for (NerfRenderProxy& proxy : _proxies) {
			proxy.field.load_snapshot();
			proxy.modifiers.copy_from_host();
			_nerf_props.emplace_back(proxy);
		}
		nerf_props.resize_and_copy_from_host(_nerf_props);
	}

	std::vector<NerfRenderProxy>& get_renderables() {
		return _proxies;
	}
};

NGP_NAMESPACE_END
