// 04_textures: sample material textures in the fragment shader.
//
// Demonstrates the sampled-image resource kind: a UV AttributeStream and two
// Sampler2D contracts (a base-color map and a packed ARM map holding ambient
// occlusion, roughness and metalness), plus the host-side work of getting
// pixels onto the GPU (staging buffer -> device-local image -> layout
// transitions) and binding each image through a descriptor like any other
// resource.
//
// Both textures are generated on the CPU, so the sample needs no image files.
// The ARM map sweeps roughness across U and metalness across V, so a single
// cube face shows the whole range the shading model responds to.
//
// The cube geometry is 02's, with a UV stream added.

#include <cstdlib>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <rcgp.hpp>

#include "app.hpp"

using namespace rcgp;

constexpr uint32_t texture_size = 256;
constexpr uint32_t checker_squares = 8;

struct View {
	float4x4 model;
	float4x4 mvp;
	float3 eye;

	$reflection(model, mvp, eye);
};

AttributeStream <float3> position;
AttributeStream <float3> normal;
AttributeStream <float2> uv;

PushConstant <View> view;
Sampler2D albedo_map;
Sampler2D arm_map;

using PositionVertex = $resource_t(position) ::element_type;
using NormalVertex = $resource_t(normal) ::element_type;
using UvVertex = $resource_t(uv) ::element_type;

using CubeIndexBuffer = IndexBuffer <Topology::eTriangleList, uint32_t>;
using CubeIndex = CubeIndexBuffer::element_type;

auto vs = $shader(vertex)(
	$contracts((p, position), (n, normal), (t, uv), (pc, view)),
	ClipPosition cpos
)
{
	cpos = pc.mvp * float4(p, 1.0f);
	float3 world_p = float3(pc.model * float4(p, 1.0f));
	float3 world_n = normalize(float3(pc.model * float4(n, 0.0f)));
	return std::tuple { Smooth(world_p), Smooth(world_n), Smooth(float2(t)) };
};

// Cook-Torrance with the roughness and metalness read from the ARM map, so the
// texture drives the highlight shape and the diffuse/specular split.
auto fs = $shader(fragment)(
	$contracts(albedo_map, arm_map, (pc, view)),
	float3 world_p,
	float3 nrm,
	float2 tex_coord
) -> float4
{
	float3 albedo = albedo_map.sample(tex_coord).xyz;
	float3 arm = arm_map.sample(tex_coord).xyz;

	f32 ao = arm.x;
	f32 roughness = max(arm.y, f32(0.05f));
	f32 metalness = arm.z;

	float3 n = normalize(nrm);
	float3 wo = normalize(pc.eye - world_p);
	float3 wi = normalize(float3(0.35f, 0.75f, 0.55f));
	float3 h = normalize(wi + wo);

	f32 n_dot_l = max(dot(n, wi), f32(0.0f));
	f32 n_dot_v = max(dot(n, wo), f32(1e-4f));
	f32 n_dot_h = max(dot(n, h), f32(0.0f));
	f32 h_dot_v = max(dot(h, wo), f32(0.0f));

	// GGX normal distribution.
	f32 a = roughness * roughness;
	f32 a2 = a * a;
	f32 denom = n_dot_h * n_dot_h * (a2 - f32(1.0f)) + f32(1.0f);
	f32 D = a2 / (f32(3.14159265f) * denom * denom);

	// Smith-Schlick geometry term.
	f32 k = a / f32(2.0f);
	f32 G = (n_dot_l / (n_dot_l * (f32(1.0f) - k) + k))
		* (n_dot_v / (n_dot_v * (f32(1.0f) - k) + k));

	// Schlick Fresnel; metals take their reflectance from the base color.
	float3 f0 = float3(0.04f) + (albedo - float3(0.04f)) * metalness;
	float3 F = f0 + (float3(1.0f) - f0) * pow(f32(1.0f) - h_dot_v, f32(5.0f));

	float3 specular = D * G * F / (f32(4.0f) * n_dot_l * n_dot_v + f32(1e-4f));
	float3 diffuse = (float3(1.0f) - F) * (f32(1.0f) - metalness) * albedo / f32(3.14159265f);

	float3 color = (diffuse + specular) * n_dot_l;
	color += 0.25f * albedo * ao;

	return float4(color, 1.0f);
};

auto new_pipeline(const Device &device, vk::Format color_format, vk::Format depth_format)
{
	return RasterizationCombinator <Topology::eTriangleList> {
		.device = device,
		.compiler = ShaderCompiler(),
		.render_state = vk::PipelineRenderingCreateInfo()
			.setColorAttachmentFormats(color_format)
			.setDepthAttachmentFormat(depth_format),
		.options = RasterizationOptions {
			.extent = {},
			.depth_test = true,
			.cull_mode = vk::CullModeFlagBits::eBack,
			.polygon_mode = vk::PolygonMode::eFill,
			.alpha_blend = false,
		},
	} (vs, fs);
}

const auto cube_vertices = std::vector <PositionVertex> {
	PositionVertex(glm::vec3(-1, -1,  1)), PositionVertex(glm::vec3( 1, -1,  1)),
	PositionVertex(glm::vec3( 1,  1,  1)), PositionVertex(glm::vec3(-1,  1,  1)),
	PositionVertex(glm::vec3( 1, -1, -1)), PositionVertex(glm::vec3(-1, -1, -1)),
	PositionVertex(glm::vec3(-1,  1, -1)), PositionVertex(glm::vec3( 1,  1, -1)),
	PositionVertex(glm::vec3(-1, -1, -1)), PositionVertex(glm::vec3(-1, -1,  1)),
	PositionVertex(glm::vec3(-1,  1,  1)), PositionVertex(glm::vec3(-1,  1, -1)),
	PositionVertex(glm::vec3( 1, -1,  1)), PositionVertex(glm::vec3( 1, -1, -1)),
	PositionVertex(glm::vec3( 1,  1, -1)), PositionVertex(glm::vec3( 1,  1,  1)),
	PositionVertex(glm::vec3(-1,  1,  1)), PositionVertex(glm::vec3( 1,  1,  1)),
	PositionVertex(glm::vec3( 1,  1, -1)), PositionVertex(glm::vec3(-1,  1, -1)),
	PositionVertex(glm::vec3(-1, -1, -1)), PositionVertex(glm::vec3( 1, -1, -1)),
	PositionVertex(glm::vec3( 1, -1,  1)), PositionVertex(glm::vec3(-1, -1,  1)),
};

const auto cube_normals = std::vector <NormalVertex> {
	NormalVertex(glm::vec3( 0,  0,  1)), NormalVertex(glm::vec3( 0,  0,  1)),
	NormalVertex(glm::vec3( 0,  0,  1)), NormalVertex(glm::vec3( 0,  0,  1)),
	NormalVertex(glm::vec3( 0,  0, -1)), NormalVertex(glm::vec3( 0,  0, -1)),
	NormalVertex(glm::vec3( 0,  0, -1)), NormalVertex(glm::vec3( 0,  0, -1)),
	NormalVertex(glm::vec3(-1,  0,  0)), NormalVertex(glm::vec3(-1,  0,  0)),
	NormalVertex(glm::vec3(-1,  0,  0)), NormalVertex(glm::vec3(-1,  0,  0)),
	NormalVertex(glm::vec3( 1,  0,  0)), NormalVertex(glm::vec3( 1,  0,  0)),
	NormalVertex(glm::vec3( 1,  0,  0)), NormalVertex(glm::vec3( 1,  0,  0)),
	NormalVertex(glm::vec3( 0,  1,  0)), NormalVertex(glm::vec3( 0,  1,  0)),
	NormalVertex(glm::vec3( 0,  1,  0)), NormalVertex(glm::vec3( 0,  1,  0)),
	NormalVertex(glm::vec3( 0, -1,  0)), NormalVertex(glm::vec3( 0, -1,  0)),
	NormalVertex(glm::vec3( 0, -1,  0)), NormalVertex(glm::vec3( 0, -1,  0)),
};

// Each face gets the full [0, 1] UV square.
auto make_cube_uvs() -> std::vector <UvVertex>
{
	auto uvs = std::vector <UvVertex> ();
	for (int face = 0; face < 6; face++) {
		uvs.push_back(UvVertex(glm::vec2(0, 0)));
		uvs.push_back(UvVertex(glm::vec2(1, 0)));
		uvs.push_back(UvVertex(glm::vec2(1, 1)));
		uvs.push_back(UvVertex(glm::vec2(0, 1)));
	}
	return uvs;
}

const auto cube_indices = std::vector <CubeIndex> {
	CubeIndex(glm::uvec3( 0,  1,  2)), CubeIndex(glm::uvec3( 0,  2,  3)),
	CubeIndex(glm::uvec3( 4,  5,  6)), CubeIndex(glm::uvec3( 4,  6,  7)),
	CubeIndex(glm::uvec3( 8,  9, 10)), CubeIndex(glm::uvec3( 8, 10, 11)),
	CubeIndex(glm::uvec3(12, 13, 14)), CubeIndex(glm::uvec3(12, 14, 15)),
	CubeIndex(glm::uvec3(16, 17, 18)), CubeIndex(glm::uvec3(16, 18, 19)),
	CubeIndex(glm::uvec3(20, 21, 22)), CubeIndex(glm::uvec3(20, 22, 23)),
};

const auto cube_index_count = uint32_t(cube_indices.size() * 3);

// Base color: a two-tone checkerboard.
auto make_albedo_pixels() -> std::vector <uint8_t>
{
	auto pixels = std::vector <uint8_t> (size_t(texture_size) * texture_size * 4);
	const uint32_t square = texture_size / checker_squares;

	for (uint32_t y = 0; y < texture_size; y++) {
		for (uint32_t x = 0; x < texture_size; x++) {
			bool odd = ((x / square) + (y / square)) % 2 == 1;
			auto tint = odd ? glm::vec3(0.95f, 0.35f, 0.25f) : glm::vec3(0.90f, 0.88f, 0.85f);

			size_t i = 4 * (size_t(y) * texture_size + x);
			pixels[i + 0] = uint8_t(255.0f * tint.x);
			pixels[i + 1] = uint8_t(255.0f * tint.y);
			pixels[i + 2] = uint8_t(255.0f * tint.z);
			pixels[i + 3] = 255;
		}
	}

	return pixels;
}

// Packed ARM: R = ambient occlusion, G = roughness, B = metalness. Roughness
// sweeps along U and metalness along V; occlusion darkens the checker seams.
auto make_arm_pixels() -> std::vector <uint8_t>
{
	auto pixels = std::vector <uint8_t> (size_t(texture_size) * texture_size * 4);
	const uint32_t square = texture_size / checker_squares;

	for (uint32_t y = 0; y < texture_size; y++) {
		for (uint32_t x = 0; x < texture_size; x++) {
			float roughness = float(x) / float(texture_size - 1);
			float metalness = float(y) / float(texture_size - 1);

			// Darken a band around each checker cell border.
			uint32_t fx = x % square;
			uint32_t fy = y % square;
			uint32_t edge = square / 8;
			bool near_edge = fx < edge or fy < edge
				or fx >= square - edge or fy >= square - edge;
			float ao = near_edge ? 0.45f : 1.0f;

			size_t i = 4 * (size_t(y) * texture_size + x);
			pixels[i + 0] = uint8_t(255.0f * ao);
			pixels[i + 1] = uint8_t(255.0f * roughness);
			pixels[i + 2] = uint8_t(255.0f * metalness);
			pixels[i + 3] = 255;
		}
	}

	return pixels;
}

const auto texture_sampler = vk::SamplerCreateInfo()
	.setMagFilter(vk::Filter::eLinear)
	.setMinFilter(vk::Filter::eLinear)
	.setAddressModeU(vk::SamplerAddressMode::eRepeat)
	.setAddressModeV(vk::SamplerAddressMode::eRepeat)
	.setAddressModeW(vk::SamplerAddressMode::eRepeat);

auto make_model(float t) -> glm::mat4
{
	auto m = glm::rotate(glm::mat4(1.0f), t, glm::vec3(0.0f, 1.0f, 0.0f));
	return glm::rotate(m, t * 0.6f, glm::vec3(1.0f, 0.0f, 0.0f));
}

const auto eye_position = glm::vec3(0, 0, 5);

auto make_mvp(const glm::mat4 &model, const vk::Extent2D &extent) -> glm::mat4
{
	const auto aspect = float(extent.width) / float(extent.height);
	auto camera = glm::lookAt(eye_position, glm::vec3(0), glm::vec3(0, 1, 0));
	auto projection = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
	projection[1][1] *= -1.0f;
	return projection * camera * model;
}

struct TexturedCube : ModuleApp <TexturedCube> {
	DescriptorPool dpool;
	$return_t(new_pipeline) pipeline;

	$resource_t(position) pos_buf;
	$resource_t(normal) nrm_buf;
	$resource_t(uv) uv_buf;
	CubeIndexBuffer idx_buf;

	Image albedo_tex;
	Image arm_tex;
	BoundDescriptor <albedo_map> albedo_desc;
	BoundDescriptor <arm_map> arm_desc;

	$resource_t(view) view_pc;

	// Allocates a device-local RGBA8 image; the caller uploads into it.
	auto new_texture() -> Image
	{
		return Image::from(device, {
			.extent = vk::Extent3D(texture_size, texture_size, 1),
			.format = vk::Format::eR8G8B8A8Unorm,
			.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			.properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
			.tiling = vk::ImageTiling::eOptimal,
			.aspect = vk::ImageAspectFlagBits::eColor,
		});
	}

	TexturedCube()
		: ModuleApp(Config { .title = "04 - Textures", .depth = true })
	{
		dpool = DescriptorPool::from(device, {
			.max_sets = 2,
			.combined_image_samplers = 2,
		});

		pipeline = new_pipeline(device, window.format, depth_format);

		auto uvs = make_cube_uvs();
		pos_buf = $resource_t(position) ::from(device, cube_vertices.size(), host_visible_coherent).write(cube_vertices);
		nrm_buf = $resource_t(normal) ::from(device, cube_normals.size(), host_visible_coherent).write(cube_normals);
		uv_buf = $resource_t(uv) ::from(device, uvs.size(), host_visible_coherent).write(uvs);
		idx_buf = CubeIndexBuffer::from(device, cube_indices.size(), host_visible_coherent).write(cube_indices);

		albedo_tex = new_texture();
		arm_tex = new_texture();

		// Pixels reach device-local memory through host-visible staging
		// buffers, with each image transitioned into the right layout on
		// either side of the copy.
		auto albedo_pixels = make_albedo_pixels();
		auto arm_pixels = make_arm_pixels();

		auto staging_of = [&](const std::vector <uint8_t> &pixels) {
			return Buffer::from(device,
				pixels.size(),
				vk::BufferUsageFlagBits::eTransferSrc,
				host_visible_coherent
			).write(std::span(pixels));
		};

		auto albedo_staging = staging_of(albedo_pixels);
		auto arm_staging = staging_of(arm_pixels);

		device.one_shot(queue, cpool, [&](auto cmd) {
			for (auto [image, staging] : {
				std::pair { &albedo_tex, &albedo_staging },
				std::pair { &arm_tex, &arm_staging },
			}) {
				cmd.transition(*image, vk::ImageLayout::eTransferDstOptimal);
				cmd.copy_buffer_to_image(*staging, *image);
				cmd.transition(*image, vk::ImageLayout::eShaderReadOnlyOptimal);
			}
		});

		albedo_staging.destroy();
		arm_staging.destroy();

		std::tie(albedo_desc, arm_desc) = device.update_descriptors(
			DescriptorWrite {
				device.new_descriptor <albedo_map> (dpool),
				MirrorSampler::from(device, texture_sampler, albedo_tex)
			},
			DescriptorWrite {
				device.new_descriptor <arm_map> (dpool),
				MirrorSampler::from(device, texture_sampler, arm_tex)
			}
		);
	}

	auto record(ModuleFrame &f)
	{
		auto model = make_model(f.elapsed);
		view_pc.model = model;
		view_pc.mvp = make_mvp(model, f.extent);
		view_pc.eye = eye_position;

		return f.rec
			| transition(&f.target, vk::ImageLayout::eColorAttachmentOptimal)
			| transition(f.depth, vk::ImageLayout::eDepthStencilAttachmentOptimal)
			| begin_rendering(f.render_area, { f.color }, f.depth_attachment)
			| set_viewport_scissor(f.extent)
			| bind_pipeline(pipeline)
			| bind_descriptors(albedo_desc, arm_desc)
			| bind_push_constants <view> (view_pc)
			| bind_vertex_buffers <position, normal, uv> (pos_buf, nrm_buf, uv_buf)
			| bind_index_buffer <Topology::eTriangleList> (idx_buf)
			| draw_indexed(cube_index_count)
			| end_rendering()
			| transition(&f.target, vk::ImageLayout::ePresentSrcKHR);
	}
};

int main()
{
	TexturedCube app;
	app.run();
}
