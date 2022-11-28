 /** @file   render_modifiers.cuh
  *
  *  @author James Perlman, avid instant-ngp fan
  */

#pragma once

#include <vector>

#include <neural-graphics-primitives/nerf/mask_3D.cuh>
#include <neural-graphics-primitives/nerf/render_modifiers_descriptor.cuh>
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

	RenderModifiers() : mask_descriptors(), masks() {};
	RenderModifiers(const std::vector<Mask3D>& cpu_masks) : mask_descriptors(cpu_masks), masks() {};
	RenderModifiers(
		const RenderModifiersDescriptor& descriptor,
		const RenderModifiersDescriptor& global_modifiers,
		const Eigen::Matrix4f& global_to_local_transform
	) {
		mask_descriptors.reserve(descriptor.masks.size() + global_modifiers.masks.size());

		for (const Mask3D& mask : descriptor.masks) {
			mask_descriptors.emplace_back(mask);
		}

		for (const Mask3D& global_mask : global_modifiers.masks) {
			Mask3D local_mask = global_mask.transformed_by(global_to_local_transform);
			mask_descriptors.emplace_back(local_mask);
		}
	}

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

/*
#include <neural-graphics-primitives/nerf/render_request.cuh>
struct RenderModifiers {
private:
	RenderRequest render_request;
public:

	tcnn::GPUMemory<Mask3D> global_masks;
	tcnn::GPUMemory<Mask3D> nerf_masks; // n per nerf, indexed by _mask_idx_by_nerf_idx
	tcnn::GPUMemory<uint32_t> nerf_masks_idx_by_nerf_idx;
	tcnn::GPUMemory<uint32_t> n_nerf_masks_by_nerf_idx;

	tcnn::GPUMemory<BoundingBox> nerf_aabbs; // ordered same as _nerfs, 1 per nerf
    
    // RenderModifiers operator=(const RenderModifiersDescriptor& descriptor) {
    //    mask_descriptors = descriptor.masks;
    //    return *this;
    // }

	RenderModifiers(const RenderRequest& render_request)
		: render_request(render_request)
		, global_masks()
		, nerf_masks()
		, nerf_masks_idx_by_nerf_idx()
		, n_nerf_masks_by_nerf_idx()
		, nerf_aabbs()
	{};
       
	// this function returns an updated copy of the input masks, but modified with an optional "All" mask at the beginning so it's compatible with the current render engine
	std::vector<Mask3D> renderable_masks(const std::vector<Mask3D>& masks) const {
		std::vector<Mask3D> result(masks);
		Mask3D first_mask = masks[0];
		if (first_mask.shape != EMaskShape::All) {
			// add another mask to beginning of m_render_masks
			EMaskMode mode = first_mask.mode == EMaskMode::Add ? EMaskMode::Subtract : EMaskMode::Add;
			result.insert(result.begin(), Mask3D::All(mode));
		}
		return result;
	}

	void copy_from_host() {
		// copy global masks
		global_masks.resize_and_copy_from_host(renderable_masks(render_request.modifiers.masks));

		// copy nerf masks

		// this can be abstracted when we introduce distortion fields and other arbitrarily-sized nerf property lists
		std::vector<Mask3D> _nerf_masks;
		std::vector<uint32_t> _nerf_masks_idx_by_nerf_idx;
		std::vector<uint32_t> _n_nerf_masks_by_nerf_idx;
		uint32_t _mask_start_idx = 0;

		for (const auto& nerf : render_request.nerfs) {
			auto masks = renderable_masks(nerf.modifiers.masks);
			for (const auto& mask : masks) {
				_nerf_masks.emplace_back(mask);
				++_mask_start_idx;
			}

			_n_nerf_masks_by_nerf_idx.emplace_back(masks.size());
			_nerf_masks_idx_by_nerf_idx.emplace_back(_mask_start_idx);
		}

		nerf_masks.resize_and_copy_from_host(_nerf_masks);
		nerf_masks_idx_by_nerf_idx.resize_and_copy_from_host(_nerf_masks_idx_by_nerf_idx);
		n_nerf_masks_by_nerf_idx.resize_and_copy_from_host(_n_nerf_masks_by_nerf_idx);
	}
};
*/
