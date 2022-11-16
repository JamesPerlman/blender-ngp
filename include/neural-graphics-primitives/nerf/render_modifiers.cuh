/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   render_modifiers.cuh
 * 
 *  @author James Perlman, avid instant-ngp fan
 */

#pragma once

#include <vector>

#include <neural-graphics-primitives/nerf/mask_3D.cuh>
#include <tiny-cuda-nn/common.h>

NGP_NAMESPACE_BEGIN

struct RenderModifiers {
private:
    std::vector<Mask3D> mask_descriptors;
    
public:
    tcnn::GPUMemory<Mask3D> masks;
    
    RenderModifiers operator=(const RenderModifiersDescriptor& descriptor) {
        mask_descriptors = descriptor.masks;
        return *this;
    }

    RenderModifiers(): mask_descriptors(), masks() {}
    RenderModifiers(const std::vector<Mask3D>& cpu_masks) : mask_descriptors(cpu_masks), masks() {}
    RenderModifiers(const RenderModifiersDescriptor& descriptor) : RenderModifiers(descriptor.masks) {}

    void copy_from_host() {
        if (mask_descriptors.size() == 0) {
            masks.resize_and_copy_from_host(mask_descriptors);
            return;
        }

        std::vector<Mask3D> render_masks(mask_descriptors);
        Mask3D first_mask = render_masks[0];
        if (first_mask.shape != EMaskShape::All) {
            // add another mask to beginning of m_render_masks
            EMaskMode mode = first_mask.mode == EMaskMode::Add ? EMaskMode::Subtract : EMaskMode::Add;
            render_masks.insert(render_masks.begin(), Mask3D::All(mode));
        }
        masks.resize_and_copy_from_host(render_masks);
    }
};

NGP_NAMESPACE_END
