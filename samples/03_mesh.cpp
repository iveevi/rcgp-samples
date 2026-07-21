// 03_mesh: load and render a glTF/GLB mesh with simple solid shading.
//
// Demonstrates: loading external geometry (load_gltf) into
// AttributeStream buffers, a PushConstant transform contract, a UniformBuffer
// material bound through a descriptor, and a command module draw. Shading is a
// single directional Lambert term over a solid albedo.
//
// Usage: 03_mesh <scene.gltf|scene.glb>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <print>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <rcgp.hpp>

#include "app.hpp"
#include "gltf.hpp"

using namespace rcgp;

struct View {
	float4x4 model;
	float4x4 mvp;

	$reflection(model, mvp);
};

struct Material {
	float3 albedo;

	$reflection(albedo);
};

AttributeStream <float3> position;
AttributeStream <float3> normal;

PushConstant <View> view;
UniformBuffer <Material> material;

using PositionVertex = $resource_t(position) ::element_type;
using NormalVertex = $resource_t(normal) ::element_type;
using IndexBufferType = IndexBuffer <Topology::eTriangleList, uint32_t>;
using TriangleIndex = IndexBufferType::element_type;

auto vs = $shader(vertex)(
	$contracts((p, position), (n, normal), (pc, view)),
	ClipPosition cpos
) -> float3
{
	cpos = pc.mvp * float4(p, 1.0f);
	return normalize(float3(pc.model * float4(n, 0.0f)));
};

auto fs = $shader(fragment)(
	$contracts(material),
	float3 nrm
) -> float4
{
	auto light_dir = normalize(float3(0.4f, 0.8f, 0.5f));
	auto lambert = max(dot(normalize(nrm), light_dir), f32(0.0f));
	auto shade = f32(0.15f) + lambert * f32(0.85f);
	return float4(material.albedo * shade, 1.0f);
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

auto make_mvp(const glm::mat4 &model, const vk::Extent2D &extent) -> glm::mat4
{
	const auto aspect = float(extent.width) / float(extent.height);
	auto camera = glm::lookAt(glm::vec3(0, 1.5, 4), glm::vec3(0), glm::vec3(0, 1, 0));
	auto projection = glm::perspective(glm::radians(55.0f), aspect, 0.1f, 100.0f);
	projection[1][1] *= -1.0f;
	return projection * camera * model;
}

const char *scene_path = nullptr;

auto load_scene() -> Mesh
{
	auto scene = load_gltf(scene_path);
	std::println("loaded {}: {} vertices, {} triangles",
		scene_path, scene.positions.size(), scene.triangles.size());
	return scene;
}

// Fit the scene into view: recenter on the bounding box and scale so the
// bounding sphere has radius ~1.5.
auto scene_fit(const Mesh &scene) -> glm::mat4
{
	const float r = std::max(scene.radius(), 1e-4f);
	return glm::scale(glm::mat4(1.0f), glm::vec3(1.5f / r))
		* glm::translate(glm::mat4(1.0f), -scene.center());
}

auto scene_positions(const Mesh &scene) -> std::vector <PositionVertex>
{
	auto positions = std::vector <PositionVertex> {};
	positions.reserve(scene.positions.size());
	for (const auto &p : scene.positions)
		positions.push_back(PositionVertex(p));
	return positions;
}

auto scene_normals(const Mesh &scene) -> std::vector <NormalVertex>
{
	auto normals = std::vector <NormalVertex> {};
	normals.reserve(scene.normals.size());
	for (const auto &n : scene.normals)
		normals.push_back(NormalVertex(n));
	return normals;
}

auto scene_indices(const Mesh &scene) -> std::vector <TriangleIndex>
{
	auto indices = std::vector <TriangleIndex> {};
	indices.reserve(scene.triangles.size());
	for (const auto &t : scene.triangles)
		indices.push_back(TriangleIndex(t));
	return indices;
}

struct MeshApp : ModuleApp <MeshApp> {
	Mesh scene;
	glm::mat4 fit;
	uint32_t index_count;

	DescriptorPool dpool;
	$return_t(new_pipeline) pipeline;

	$resource_t(position) pos_buf;
	$resource_t(normal) nrm_buf;
	IndexBufferType idx_buf;
	$resource_t(material) mat_buf;
	BoundDescriptor <material> mat_desc;

	$resource_t(view) view_pc;

	MeshApp()
		: ModuleApp(Config { .title = "03 - Mesh", .depth = true, .clear = { 0.02f, 0.02f, 0.03f, 1.0f } })
	{
		scene = load_scene();
		fit = scene_fit(scene);

		auto positions = scene_positions(scene);
		auto normals = scene_normals(scene);
		auto indices = scene_indices(scene);
		index_count = uint32_t(indices.size() * 3);

		dpool = DescriptorPool::from(device, {
			.max_sets = 1,
			.uniform_buffers = 1,
		});

		pipeline = new_pipeline(device, window.format, depth_format);

		pos_buf = $resource_t(position) ::from(device, positions.size(), host_visible_coherent).write(positions);
		nrm_buf = $resource_t(normal) ::from(device, normals.size(), host_visible_coherent).write(normals);
		idx_buf = IndexBufferType::from(device, indices.size(), host_visible_coherent).write(indices);

		mat_buf = $resource_t(material) ::from(device, host_visible_coherent)
			.write({ .albedo = glm::vec3(0.82, 0.80, 0.76) });

		mat_desc = device.update_descriptors(
			DescriptorWrite { device.new_descriptor <material> (dpool), mat_buf }
		);
	}

	auto record(ModuleFrame &f)
	{
		auto model = glm::rotate(glm::mat4(1.0f), f.elapsed * 0.4f, glm::vec3(0, 1, 0)) * fit;
		view_pc.model = model;
		view_pc.mvp = make_mvp(model, f.extent);

		return f.rec
			| transition(&f.target, vk::ImageLayout::eColorAttachmentOptimal)
			| transition(f.depth, vk::ImageLayout::eDepthStencilAttachmentOptimal)
			| begin_rendering(f.render_area, { f.color }, f.depth_attachment)
			| set_viewport_scissor(f.extent)
			| bind_pipeline(pipeline)
			| bind_descriptors(mat_desc)
			| bind_push_constants <view> (view_pc)
			| bind_vertex_buffers <position, normal> (pos_buf, nrm_buf)
			| bind_index_buffer <Topology::eTriangleList> (idx_buf)
			| draw_indexed(index_count)
			| end_rendering()
			| transition(&f.target, vk::ImageLayout::ePresentSrcKHR);
	}
};

int main(int argc, char **argv)
{
	if (argc < 2) {
		std::println(std::cerr, "usage: {} <scene.gltf|scene.glb>", argv[0]);
		return EXIT_FAILURE;
	}

	scene_path = argv[1];

	MeshApp app;
	app.run();
}
