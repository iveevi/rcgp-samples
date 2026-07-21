// 01_triangle: the smallest RCGP raster program.
//
// Demonstrates: an AttributeStream vertex contract, a vertex/fragment shader
// pair written in the embedded DSL, a RasterizationCombinator, and a frame
// recorded as a command module (the `|` pipeline). Session, device, window and
// the frame loop come from the App base; the sample derives from it and
// implements record(). No depth, no descriptors, no push constants.

#include <vector>

#include <glm/glm.hpp>

#include <rcgp.hpp>

#include "app.hpp"

using namespace rcgp;

AttributeStream <float3> position;

using PositionVertex = $resource_t(position) ::element_type;

const auto triangle_vertices = std::vector <PositionVertex> {
	PositionVertex(glm::vec3( 0.0f,  0.6f, 0.0f)),
	PositionVertex(glm::vec3(-0.6f, -0.6f, 0.0f)),
	PositionVertex(glm::vec3( 0.6f, -0.6f, 0.0f)),
};

auto vs = $shader(vertex)(
	$contracts((p, position)),
	ClipPosition cpos
) -> float3
{
	cpos = float4(p, 1.0f);
	return float3(0.5f) + p * 0.5f;
};

auto fs = $shader(fragment)(
	float3 color
)
{
	return float4(color, 1.0f);
};


auto new_pipeline(const Device &device, vk::Format color_format)
{
	return RasterizationCombinator <Topology::eTriangleList> {
		.device = device,
		.compiler = ShaderCompiler(),
		.render_state = vk::PipelineRenderingCreateInfo()
			.setColorAttachmentFormats(color_format),
		.options = RasterizationOptions {
			.extent = {},
			.depth_test = false,
			.cull_mode = vk::CullModeFlagBits::eNone,
			.polygon_mode = vk::PolygonMode::eFill,
			.alpha_blend = false,
		},
	} (vs, fs);
}

struct Triangle : ModuleApp <Triangle> {
	$return_t(new_pipeline) pipeline;
	$resource_t(position) pos_buf;

	Triangle()
		: ModuleApp(Config { .title = "01 - Triangle" })
	{
		pipeline = new_pipeline(device, window.format);
		pos_buf = $resource_t(position) ::from(device, triangle_vertices.size(), host_visible_coherent)
			.write(triangle_vertices);
	}

	auto record(ModuleFrame &f)
	{
		return f.rec
			| transition(&f.target, vk::ImageLayout::eColorAttachmentOptimal)
			| begin_rendering(f.render_area, { f.color })
			| set_viewport_scissor(f.extent)
			| bind_pipeline(pipeline)
			| bind_vertex_buffers <position> (pos_buf)
			| draw(uint32_t(triangle_vertices.size()))
			| end_rendering()
			| transition(&f.target, vk::ImageLayout::ePresentSrcKHR);
	}
};

int main()
{
	Triangle app;
	app.run();
}
