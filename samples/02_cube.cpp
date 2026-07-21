// 02_cube: an indexed, depth-tested, animated mesh.
//
// Demonstrates: multiple AttributeStreams (position + normal), a PushConstant
// contract for per-frame transforms, a UniformBuffer contract bound through a
// descriptor, an IndexBuffer, depth attachment setup, and a frame recorded as
// a command module. The command module statically enforces that every resource
// the pipeline depends on is bound before the draw.

#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <rcgp.hpp>

#include "app.hpp"

using namespace rcgp;

auto make_model(float t) -> glm::mat4
{
	auto m = glm::rotate(glm::mat4(1.0f), t, glm::vec3(0.0f, 1.0f, 0.0f));
	return glm::rotate(m, t * 0.6f, glm::vec3(1.0f, 0.0f, 0.0f));
}

auto make_mvp(const glm::mat4 &model, const vk::Extent2D &extent) -> glm::mat4
{
	const auto aspect = float(extent.width) / float(extent.height);
	auto view = glm::lookAt(
		glm::vec3(0.0f, 0.0f, 5.0f),
		glm::vec3(0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f)
	);
	auto projection = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
	projection[1][1] *= -1.0f;
	return projection * view * model;
}

AttributeStream <float3> position;
AttributeStream <float3> normal;

using PositionVertex = $resource_t(position) ::element_type;
using NormalVertex = $resource_t(normal) ::element_type;

using CubeIndexBuffer = IndexBuffer <Topology::eTriangleList, uint32_t>;
using CubeIndex = CubeIndexBuffer::element_type;

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

const auto cube_indices = std::vector <CubeIndex> {
	CubeIndex(glm::uvec3( 0,  1,  2)), CubeIndex(glm::uvec3( 0,  2,  3)),
	CubeIndex(glm::uvec3( 4,  5,  6)), CubeIndex(glm::uvec3( 4,  6,  7)),
	CubeIndex(glm::uvec3( 8,  9, 10)), CubeIndex(glm::uvec3( 8, 10, 11)),
	CubeIndex(glm::uvec3(12, 13, 14)), CubeIndex(glm::uvec3(12, 14, 15)),
	CubeIndex(glm::uvec3(16, 17, 18)), CubeIndex(glm::uvec3(16, 18, 19)),
	CubeIndex(glm::uvec3(20, 21, 22)), CubeIndex(glm::uvec3(20, 22, 23)),
};

const auto cube_index_count = uint32_t(cube_indices.size() * 3);

struct View {
	float4x4 model;
	float4x4 mvp;

	$reflection(model, mvp);
};

PushConstant <View> view;

auto vs = $shader(vertex)(
	$contracts((p, position), (n, normal), (pc, view)),
	ClipPosition cpos
) -> float3
{
	cpos = pc.mvp * float4(p, 1.0f);
	return normalize(float3(pc.model * float4(n, 0.0f)));
};

struct Material {
	float3 albedo;

	$reflection(albedo);
};

UniformBuffer <Material> material;

auto fs = $shader(fragment)(
	$contracts(material),
	float3 nrm
)
{
	auto light_dir = normalize(float3(0.35f, 0.75f, 0.55f));
	auto lambert = max(dot(normalize(nrm), light_dir), f32(0.0f));
	auto diffuse = f32(0.18f) + lambert * f32(0.82f);
	return float4(material.albedo * diffuse, 1.0f);
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

struct Cube : ModuleApp <Cube> {
	DescriptorPool dpool;
	$return_t(new_pipeline) pipeline;

	$resource_t(position) pos_buf;
	$resource_t(normal) nrm_buf;
	CubeIndexBuffer idx_buf;
	$resource_t(material) mat_buf;
	BoundDescriptor <material> mat_desc;

	$resource_t(view) view_pc;

	Cube()
		: ModuleApp(Config { .title = "02 - Cube", .depth = true })
	{
		dpool = DescriptorPool::from(device, {
			.max_sets = 1,
			.uniform_buffers = 1,
		});

		pipeline = new_pipeline(device, window.format, depth_format);

		pos_buf = $resource_t(position) ::from(device, cube_vertices.size(), host_visible_coherent).write(cube_vertices);
		nrm_buf = $resource_t(normal) ::from(device, cube_normals.size(), host_visible_coherent).write(cube_normals);
		idx_buf = CubeIndexBuffer::from(device, cube_indices.size(), host_visible_coherent).write(cube_indices);

		mat_buf = $resource_t(material) ::from(device, host_visible_coherent)
			.write({ .albedo = glm::vec3(0.95, 0.25, 0.18) });

		mat_desc = device.update_descriptors(
			DescriptorWrite { device.new_descriptor <material> (dpool), mat_buf }
		);
	}

	auto record(ModuleFrame &f)
	{
		auto model = make_model(f.elapsed);
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
			| draw_indexed(cube_index_count)
			| end_rendering()
			| transition(&f.target, vk::ImageLayout::ePresentSrcKHR);
	}
};

int main()
{
	Cube app;
	app.run();
}
