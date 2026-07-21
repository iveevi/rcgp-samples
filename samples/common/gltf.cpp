#include <cstring>
#include <filesystem>
#include <limits>
#include <print>
#include <stdexcept>

#include <glm/gtc/type_ptr.hpp>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "gltf.hpp"

static const cgltf_attribute *find_attribute(const cgltf_primitive &prim, cgltf_attribute_type type)
{
	for (cgltf_size i = 0; i < prim.attributes_count; i++) {
		if (prim.attributes[i].type == type)
			return &prim.attributes[i];
	}
	return nullptr;
}

// Texture keys are stable names shared between the material records and the
// decoded texture map. glTF images may be unnamed, so fall back to the index.
static std::string texture_key(const cgltf_data *data, const cgltf_texture *texture)
{
	if (not texture or not texture->image)
		return "";

	const cgltf_image *image = texture->image;
	if (image->uri and std::strncmp(image->uri, "data:", 5) != 0)
		return image->uri;

	return "image_" + std::to_string(size_t(image - data->images));
}

static bool decode_image(
	const cgltf_options &options,
	const cgltf_image &image,
	const std::filesystem::path &base_dir,
	RawTexture &out
)
{
	int width = 0;
	int height = 0;
	int channels = 0;
	uint8_t *decoded = nullptr;

	if (image.buffer_view) {
		const uint8_t *bytes = cgltf_buffer_view_data(image.buffer_view);
		if (not bytes)
			return false;

		decoded = stbi_load_from_memory(bytes, int(image.buffer_view->size),
			&width, &height, &channels, 4);
	} else if (image.uri and std::strncmp(image.uri, "data:", 5) == 0) {
		const char *comma = std::strchr(image.uri, ',');
		if (not comma)
			return false;

		void *raw = nullptr;
		auto size = cgltf_size(std::strlen(comma + 1));
		if (cgltf_load_buffer_base64(&options, size, comma + 1, &raw) != cgltf_result_success)
			return false;

		decoded = stbi_load_from_memory((const uint8_t *) raw, int(size),
			&width, &height, &channels, 4);
		std::free(raw);
	} else if (image.uri) {
		auto path = base_dir / image.uri;
		decoded = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
	}

	if (not decoded)
		return false;

	out.width = uint32_t(width);
	out.height = uint32_t(height);
	out.pixels.assign(decoded, decoded + size_t(width) * size_t(height) * 4);
	stbi_image_free(decoded);

	return true;
}

Mesh load_gltf(const std::string &path)
{
	cgltf_options options {};
	cgltf_data *data = nullptr;

	if (cgltf_parse_file(&options, path.c_str(), &data) != cgltf_result_success)
		throw std::runtime_error("cgltf: failed to parse " + path);

	if (cgltf_load_buffers(&options, data, path.c_str()) != cgltf_result_success) {
		cgltf_free(data);
		throw std::runtime_error("cgltf: failed to load buffers for " + path);
	}

	Mesh mesh;
	glm::vec3 lo(std::numeric_limits <float> ::max());
	glm::vec3 hi(std::numeric_limits <float> ::lowest());

	auto base_dir = std::filesystem::path(path).parent_path();

	for (cgltf_size i = 0; i < data->images_count; i++) {
		const cgltf_image &image = data->images[i];
		auto key = (image.uri and std::strncmp(image.uri, "data:", 5) != 0)
			? std::string(image.uri)
			: "image_" + std::to_string(i);

		RawTexture texture;
		if (decode_image(options, image, base_dir, texture))
			mesh.textures.emplace(key, std::move(texture));
		else
			std::println(stderr, "gltf: failed to decode image {}", key);
	}

	for (cgltf_size i = 0; i < data->materials_count; i++) {
		const cgltf_material &m = data->materials[i];
		MeshMaterial material;

		if (m.has_pbr_metallic_roughness) {
			const auto &pbr = m.pbr_metallic_roughness;
			material.base_color = glm::make_vec3(pbr.base_color_factor);
			material.metallic = pbr.metallic_factor;
			material.roughness = pbr.roughness_factor;
			material.albedo = texture_key(data, pbr.base_color_texture.texture);
			material.arm = texture_key(data, pbr.metallic_roughness_texture.texture);
		}

		material.emissive = glm::make_vec3(m.emissive_factor);
		material.alpha_cutoff = m.alpha_cutoff;

		mesh.materials.push_back(std::move(material));
	}

	// Primitives without a material still need a slot to point at.
	const auto default_material = uint32_t(mesh.materials.size());
	mesh.materials.push_back(MeshMaterial());

	// Triangles are bucketed by material so each bucket becomes one submesh.
	std::map <uint32_t, std::vector <glm::uvec3>> buckets;

	for (cgltf_size n = 0; n < data->nodes_count; n++) {
		const cgltf_node &node = data->nodes[n];
		if (not node.mesh)
			continue;

		float world[16];
		cgltf_node_transform_world(&node, world);
		glm::mat4 model = glm::make_mat4(world);
		glm::mat3 normal_matrix = glm::transpose(glm::inverse(glm::mat3(model)));

		for (cgltf_size p = 0; p < node.mesh->primitives_count; p++) {
			const cgltf_primitive &prim = node.mesh->primitives[p];
			if (prim.type != cgltf_primitive_type_triangles)
				continue;

			const cgltf_attribute *pos_attr = find_attribute(prim, cgltf_attribute_type_position);
			if (not pos_attr)
				continue;

			const cgltf_attribute *nrm_attr = find_attribute(prim, cgltf_attribute_type_normal);
			const cgltf_attribute *uv_attr = find_attribute(prim, cgltf_attribute_type_texcoord);
			const auto base = uint32_t(mesh.positions.size());
			const cgltf_size vcount = pos_attr->data->count;

			for (cgltf_size v = 0; v < vcount; v++) {
				glm::vec3 pos(0.0f);
				cgltf_accessor_read_float(pos_attr->data, v, glm::value_ptr(pos), 3);
				pos = glm::vec3(model * glm::vec4(pos, 1.0f));

				glm::vec3 nrm(0.0f);
				if (nrm_attr) {
					cgltf_accessor_read_float(nrm_attr->data, v, glm::value_ptr(nrm), 3);
					nrm = glm::normalize(normal_matrix * nrm);
				}

				glm::vec2 texcoord(0.0f);
				if (uv_attr)
					cgltf_accessor_read_float(uv_attr->data, v, glm::value_ptr(texcoord), 2);

				mesh.positions.push_back(pos);
				mesh.normals.push_back(nrm);
				mesh.uvs.push_back(texcoord);
				lo = glm::min(lo, pos);
				hi = glm::max(hi, pos);
			}

			auto material = prim.material
				? uint32_t(prim.material - data->materials)
				: default_material;

			auto &bucket = buckets[material];
			const auto first_triangle = mesh.triangles.size();

			if (prim.indices) {
				const cgltf_size icount = prim.indices->count;
				for (cgltf_size i = 0; i + 2 < icount; i += 3) {
					mesh.triangles.push_back(glm::uvec3(
						base + uint32_t(cgltf_accessor_read_index(prim.indices, i + 0)),
						base + uint32_t(cgltf_accessor_read_index(prim.indices, i + 1)),
						base + uint32_t(cgltf_accessor_read_index(prim.indices, i + 2))
					));
				}
			} else {
				for (cgltf_size i = 0; i + 2 < vcount; i += 3)
					mesh.triangles.push_back(glm::uvec3(base + i, base + i + 1, base + i + 2));
			}

			if (not nrm_attr) {
				for (size_t t = first_triangle; t < mesh.triangles.size(); t++) {
					const auto &tri = mesh.triangles[t];
					glm::vec3 a = mesh.positions[tri.x];
					glm::vec3 b = mesh.positions[tri.y];
					glm::vec3 c = mesh.positions[tri.z];
					glm::vec3 fn = glm::normalize(glm::cross(b - a, c - a));
					mesh.normals[tri.x] = fn;
					mesh.normals[tri.y] = fn;
					mesh.normals[tri.z] = fn;
				}
			}

			bucket.insert(bucket.end(),
				mesh.triangles.begin() + first_triangle, mesh.triangles.end());
		}
	}

	cgltf_free(data);

	if (mesh.positions.empty())
		throw std::runtime_error("cgltf: no triangle geometry in " + path);

	for (auto &[material, triangles] : buckets)
		mesh.submeshes.push_back(Submesh { std::move(triangles), material });

	mesh.bounds_min = lo;
	mesh.bounds_max = hi;
	return mesh;
}
