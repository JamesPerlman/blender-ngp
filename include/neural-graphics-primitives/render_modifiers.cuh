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

#include <neural-graphics-primitives/mask_shapes.cuh>
#include <tiny-cuda-nn/common.h>

NGP_NAMESPACE_BEGIN

struct RenderModifiers {
    std::vector<Mask3D> masks;
    tcnn::GPUMemory<Mask3D> masks_gpu;
    
    RenderModifiers operator=(const RenderModifiers& other) {
        masks = other.masks;
        return *this;
    }

    RenderModifiers(const std::vector<Mask3D>& masks) : masks(masks) {}

    RenderModifiers() {}

    void copy_from_host() {
        if (masks.size() == 0) {
            masks_gpu.resize_and_copy_from_host(masks);
            return;
        }

        std::vector<Mask3D> render_masks(masks);
        Mask3D first_mask = masks[0];
        if (first_mask.shape != EMaskShape::All) {
            // add another mask to beginning of m_render_masks
            EMaskMode mode = first_mask.mode == EMaskMode::Add ? EMaskMode::Subtract : EMaskMode::Add;
            render_masks.insert(render_masks.begin(), Mask3D::All(mode));
        }
        masks_gpu.resize_and_copy_from_host(render_masks);
    }
};

NGP_NAMESPACE_END
