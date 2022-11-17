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
#include <neural-graphics-primitives/nerf/neural_radiance_field.cuh>
#include <neural-graphics-primitives/nerf/render_request.cuh>
#include <neural-graphics-primitives/nerf/render_modifiers.cuh>

NGP_NAMESPACE_BEGIN

struct RenderData {
    RenderOutputProperties output;
    RenderCameraProperties camera;
    RenderModifiers modifiers;
    std::vector<NeuralRadianceField> nerfs;
    BoundingBox aabb;

    RenderData() {};

    void update(const RenderRequest& request) {
        output = request.output;
        camera = request.camera;
        aabb = request.aabb;
        update_modifiers(request.modifiers);
        update_nerfs(request.nerfs);
    }

    void update_modifiers(const RenderModifiersDescriptor& descriptor) {
        // todo: improve this
        modifiers = RenderModifiers(descriptor);
    }

    void update_nerfs(const std::vector<NerfDescriptor>& descriptors) {
        // delete if nerf has no matching descriptor
        static_cast<void>( // discard result from remove_if iterator
            std::remove_if(nerfs.begin(), nerfs.end(), [&](const NeuralRadianceField& nerf) {
                auto result = std::find_if(descriptors.begin(), descriptors.end(), [&](const NerfDescriptor& descriptor) {
                    return descriptor.snapshot_path == nerf.snapshot_path;
                });
                // if result was not found, remove this nerf
                return result == descriptors.end();
            })
        );
        // todo: update nerfs

        // add new nerfs
        for (const NerfDescriptor& desc : descriptors) {
            auto result = std::find_if(nerfs.begin(), nerfs.end(), [&](const NeuralRadianceField& nerf) {
                return nerf.snapshot_path == desc.snapshot_path;
            });
            if (result == nerfs.end()) {
                nerfs.emplace_back(desc);
            }
        }
    }

    void load_nerfs() {
        for (NeuralRadianceField& nerf : nerfs) {
            nerf.load_snapshot();
			nerf.modifiers.copy_from_host();
            // nerf.inference.enlarge_workspace(...); ?
        }
    }
};

NGP_NAMESPACE_END
