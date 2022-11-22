
#pragma once

#include <vector>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf/mask_3D.cuh>

NGP_NAMESPACE_BEGIN

struct RenderModifiersDescriptor {
    std::vector<Mask3D> masks;
    // todo:
    // std::vector<DistortionField> distortion_fields;
};

NGP_NAMESPACE_END