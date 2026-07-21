#pragma once

#include <map>
#include <span>
#include <string>
#include <vector>

#include <rcgp.hpp>

#include "gltf.hpp"

using namespace rcgp;

// Which channels of a material are backed by a texture. Every material binds
// both samplers, so an untextured channel points at a 1x1 fallback and is
// masked out by these flags rather than by a separate pipeline.
enum TextureFlags : int32_t {
	eAlbedoTexture = 0b1,
	eArmTexture = 0b10,
};

struct MaterialDescription {
	float3 albedo;
	float3 emissive;
	f32 metalness;
	f32 roughness;
	i32 texture_flags;

	$reflection(albedo, emissive, metalness, roughness, texture_flags);
};

// One descriptor set per material: the scalar factors plus the two maps.
struct MaterialParameters {
	UniformBuffer <MaterialDescription> desc;
	Sampler2D albedo;
	Sampler2D arm;

	$reflection(desc, albedo, arm);
};

inline ResourceGroup <MaterialParameters> material;

struct MaterialHandle {
	$resource_t(material) data;
	BoundDescriptor <material> bound;
};

inline float3 get_albedo(auto mat, float2 uv)
{
	float3 albedo = mat.desc.albedo;

	$if ((mat.desc.texture_flags & i32(eAlbedoTexture)) != i32(0)) {
		albedo = albedo * mat.albedo.sample(uv).xyz;
	};

	return albedo;
}

// Returns (ambient occlusion, roughness, metalness), matching glTF's
// occlusion-roughness-metallic channel packing.
inline float3 get_arm(auto mat, float2 uv)
{
	float3 arm = float3(1.0f, mat.desc.roughness, mat.desc.metalness);

	$if ((mat.desc.texture_flags & i32(eArmTexture)) != i32(0)) {
		float3 sampled = mat.arm.sample(uv).xyz;
		arm = float3(sampled.x, sampled.y * mat.desc.roughness, sampled.z * mat.desc.metalness);
	};

	return arm;
}

inline const auto material_sampler = vk::SamplerCreateInfo()
	.setMagFilter(vk::Filter::eLinear)
	.setMinFilter(vk::Filter::eLinear)
	.setAddressModeU(vk::SamplerAddressMode::eRepeat)
	.setAddressModeV(vk::SamplerAddressMode::eRepeat)
	.setAddressModeW(vk::SamplerAddressMode::eRepeat);

// Uploads every decoded texture to device-local memory, keyed as in the mesh.
// The map always holds a 1x1 white image under the empty key, which untextured
// material channels bind to.
inline auto upload_textures(
	const Device &device,
	const Queue &queue,
	const CommandPool &cpool,
	const Mesh &mesh
) -> std::map <std::string, Image>
{
	auto images = std::map <std::string, Image> ();
	auto staging_buffers = std::vector <Buffer> ();

	auto host_flags = vk::MemoryPropertyFlagBits::eHostVisible
		| vk::MemoryPropertyFlagBits::eHostCoherent;

	device.one_shot(queue, cpool, [&](auto cmd) {
		auto upload = [&](const std::string &key, const RawTexture &tex) {
			auto image = Image::from(device, {
				.extent = vk::Extent3D(tex.width, tex.height, 1),
				.format = vk::Format::eR8G8B8A8Unorm,
				.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
				.properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
				.tiling = vk::ImageTiling::eOptimal,
				.aspect = vk::ImageAspectFlagBits::eColor,
			});

			auto staging = Buffer::from(device,
				tex.pixels.size(),
				vk::BufferUsageFlagBits::eTransferSrc,
				host_flags
			).write(std::span(tex.pixels));

			cmd.transition(image, vk::ImageLayout::eTransferDstOptimal);
			cmd.copy_buffer_to_image(staging, image);
			cmd.transition(image, vk::ImageLayout::eShaderReadOnlyOptimal);

			images.emplace(key, image);
			staging_buffers.push_back(staging);
		};

		upload("", RawTexture { .width = 1, .height = 1, .pixels = { 255, 255, 255, 255 } });

		for (const auto &[key, tex] : mesh.textures)
			upload(key, tex);
	});

	for (auto &buffer : staging_buffers)
		buffer.destroy();

	return images;
}

// Builds one descriptor set per material, pointing at the uploaded images.
inline auto build_materials(
	const Device &device,
	const DescriptorPool &dpool,
	const std::map <std::string, Image> &images,
	const Mesh &mesh
) -> std::vector <MaterialHandle>
{
	auto sampler_for = [&](const std::string &key) {
		auto it = images.find(key);
		const auto &image = (it != images.end()) ? it->second : images.at("");
		return MirrorSampler::from(device, material_sampler, image);
	};

	auto host_flags = vk::MemoryPropertyFlagBits::eHostVisible
		| vk::MemoryPropertyFlagBits::eHostCoherent;

	auto handles = std::vector <MaterialHandle> ();

	for (const auto &m : mesh.materials) {
		int32_t flags = 0;
		if (not m.albedo.empty()) flags |= eAlbedoTexture;
		if (not m.arm.empty()) flags |= eArmTexture;

		auto desc = $resource_t(material.desc) ::from(device, host_flags).write({
			.albedo = m.base_color,
			.emissive = m.emissive,
			.metalness = m.metallic,
			.roughness = m.roughness,
			.texture_flags = flags,
		});

		auto data = $resource_t(material) {
			.desc = desc,
			.albedo = sampler_for(m.albedo),
			.arm = sampler_for(m.arm),
		};

		auto bound = device.update_descriptors(
			DescriptorWrite { device.new_descriptor <material> (dpool), data });

		handles.push_back(MaterialHandle { data, bound });
	}

	return handles;
}
