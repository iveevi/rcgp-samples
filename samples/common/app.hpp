#pragma once

#include <array>
#include <chrono>
#include <concepts>
#include <cstdlib>
#include <iostream>
#include <print>
#include <utility>

#include <vulkan/vulkan.hpp>

#include <rcgp.hpp>
#include <rcgp-ex.hpp>

template <typename Rec>
struct FrameContext {
	Rec &rec;
	rcgp::ColorTargetImage &target;
	rcgp::Image *depth;
	vk::Extent2D extent;
	vk::RenderingAttachmentInfo color;
	vk::RenderingAttachmentInfo depth_attachment;
	vk::Rect2D render_area;
	float elapsed;
};

using ModuleFrame = FrameContext <rcgp::CommandBuffer>;
using StreamerFrame = FrameContext <rcgp::CommandStream>;

template <typename Derived, typename Rec>
struct AppBase {
	struct Config {
		const char *title = "rcgp";
		uint32_t width = 1280;
		uint32_t height = 720;
		bool depth = false;
		bool mesh_shading = false;
		bool raytracing = false;
		std::array <float, 4> clear = { 0.04f, 0.06f, 0.10f, 1.0f };
	};

	static constexpr vk::Format depth_format = vk::Format::eD32Sfloat;
	static constexpr vk::MemoryPropertyFlags host_visible_coherent =
		vk::MemoryPropertyFlagBits::eHostVisible
		| vk::MemoryPropertyFlagBits::eHostCoherent;

	rcgp::Session session;
	vk::detail::DispatchLoaderDynamic dld;
	rcgp::Device device;
	rcgp::Window window;
	rcgp::Queue queue;
	rcgp::CommandPool cpool;
	Config config;

	AppBase(const Config &config)
		: config(config)
	{
		auto [created_session, loader] = rcgp::Session::from({});
		session = created_session;
		dld = loader;

		auto vk12 = vk::PhysicalDeviceVulkan12Features()
			.setScalarBlockLayout(true)
			.setBufferDeviceAddress(true)
			.setDescriptorIndexing(true)
			.setRuntimeDescriptorArray(true)
			.setShaderSampledImageArrayNonUniformIndexing(true)
			.setShaderStorageBufferArrayNonUniformIndexing(true);

		auto vk13 = vk::PhysicalDeviceVulkan13Features()
			.setDynamicRendering(true)
			.setSynchronization2(true)
			.setMaintenance4(true);

		auto mesh_features = vk::PhysicalDeviceMeshShaderFeaturesEXT()
			.setMeshShader(true)
			.setTaskShader(true);

		auto rt_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR()
			.setRayTracingPipeline(true);

		auto accel_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR()
			.setAccelerationStructure(true);

		// Manual pNext chaining so that the mesh-shading and raytracing feature
		// structs are only linked in (and their extensions enabled) when a
		// sample asks for them.
		auto extensions = std::vector <const char *> { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
		void *chain = &vk13;
		vk12.pNext = chain;
		chain = &vk12;

		if (config.mesh_shading) {
			extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
			mesh_features.pNext = chain;
			chain = &mesh_features;
		}

		if (config.raytracing) {
			extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
			extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
			extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
			rt_features.pNext = chain;
			accel_features.pNext = &rt_features;
			chain = &accel_features;
		}

		device = rcgp::Device::from(session, dld, {
			.extensions = extensions,
			.queues = { { "primary", vk::QueueFlagBits::eGraphics } },
			.pNext = chain,
		});

		window = rcgp::Window::from(session, device, {
			.width = config.width,
			.height = config.height,
			.title = config.title,
		});

		queue = device.queues.at("primary");
		cpool = rcgp::CommandPool::from(device, queue);
	}

	auto make_recorders(size_t count)
	{
		if constexpr (std::same_as <Rec, rcgp::CommandBuffer>)
			return device.new_command_buffers(cpool, count);
		else
			return device.new_command_streams(cpool, count);
	}

	void run()
	{
		auto &self = static_cast <Derived &> (*this);

		auto sync = rcgp::PresentationSynchronizer::from(device, window.images.size());
		auto recorders = make_recorders(window.images.size());

		rcgp::Image depth;
		if (config.depth) {
			auto extent = window.extent();
			depth = rcgp::Image::from(device, {
				.extent = vk::Extent3D(extent.width, extent.height, 1),
				.format = depth_format,
				.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
				.properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
				.aspect = vk::ImageAspectFlagBits::eDepth,
			});
		}

		size_t slot = 0;
		auto epoch = std::chrono::steady_clock::now();

		while (window.alive()) {
			window.poll();
			if (window.is_pressed(rcgp::Key::Q))
				window.close();

			auto img = device.acquire_next_frame(window, sync);
			slot = (slot + 1) % recorders.size();
			if (not img)
				continue;

			auto &target = window.images[*img];

			auto color_attachment = vk::RenderingAttachmentInfo()
				.setImageView(target.view)
				.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
				.setLoadOp(vk::AttachmentLoadOp::eClear)
				.setStoreOp(vk::AttachmentStoreOp::eStore)
				.setClearValue(vk::ClearValue(vk::ClearColorValue(config.clear)));

			auto depth_attachment = vk::RenderingAttachmentInfo()
				.setImageView(config.depth ? depth.view : vk::ImageView())
				.setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
				.setLoadOp(vk::AttachmentLoadOp::eClear)
				.setStoreOp(vk::AttachmentStoreOp::eDontCare)
				.setClearValue(vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0)));

			auto render_area = vk::Rect2D()
				.setOffset(vk::Offset2D(0, 0))
				.setExtent(window.extent());

			auto elapsed = std::chrono::duration <float> (
				std::chrono::steady_clock::now() - epoch
			).count();

			auto frame = FrameContext <Rec> {
				.rec = recorders[slot],
				.target = target,
				.depth = config.depth ? &depth : nullptr,
				.extent = window.extent(),
				.color = color_attachment,
				.depth_attachment = depth_attachment,
				.render_area = render_area,
				.elapsed = elapsed,
			};

			auto submittable = self.record(frame);

			queue.submit(
				submittable,
				sync.acquire_semaphore(),
				sync.render_semaphore(),
				vk::PipelineStageFlagBits::eColorAttachmentOutput,
				sync.fence()
			);

			if (not queue.present(window, *img, sync.render_semaphore())) {
				std::println(std::cerr, "presentKHR failed");
				std::abort();
			}
		}
	}
};

template <typename Derived>
struct ModuleApp : AppBase <Derived, rcgp::CommandBuffer> {
	using AppBase <Derived, rcgp::CommandBuffer> ::AppBase;
};

template <typename Derived>
struct StreamerApp : AppBase <Derived, rcgp::CommandStream> {
	using AppBase <Derived, rcgp::CommandStream> ::AppBase;
};
