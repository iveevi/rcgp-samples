#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <glm/glm.hpp>

// Decoded 8-bit RGBA image data, keyed by name in Mesh::textures.
struct RawTexture {
	uint32_t width = 0;
	uint32_t height = 0;
	std::vector <uint8_t> pixels;
};

// A metallic-roughness material. The two texture fields are keys into
// Mesh::textures; an empty key means the channel has no texture and the scalar
// factors are used on their own. `arm` is glTF's metallic-roughness image,
// which packs occlusion/roughness/metalness into R/G/B.
struct MeshMaterial {
	glm::vec3 base_color { 1.0f };
	glm::vec3 emissive { 0.0f };
	float metallic = 1.0f;
	float roughness = 1.0f;
	float alpha_cutoff = 0.5f;

	std::string albedo;
	std::string arm;
};

// The triangles of one material, indexing into the mesh's shared vertex
// streams. Drawn as its own indexed draw with that material's descriptor bound.
struct Submesh {
	std::vector <glm::uvec3> triangles;
	uint32_t material = 0;
};

// A flat, triangulated mesh loaded from a glTF/GLB file: all primitives of all
// meshes are merged into one position/normal/uv stream, with node transforms
// baked into the vertices. `triangles` is every triangle, for samples that draw
// the whole thing in one call; `submeshes` splits the same triangles by
// material, for samples that bind material textures per draw.
struct Mesh {
	std::vector <glm::vec3> positions;
	std::vector <glm::vec3> normals;
	std::vector <glm::vec2> uvs;
	std::vector <glm::uvec3> triangles;

	std::vector <Submesh> submeshes;
	std::vector <MeshMaterial> materials;
	std::map <std::string, RawTexture> textures;

	glm::vec3 bounds_min { 0.0f };
	glm::vec3 bounds_max { 0.0f };

	glm::vec3 center() const { return 0.5f * (bounds_min + bounds_max); }
	float radius() const { return 0.5f * glm::length(bounds_max - bounds_min); }
};

// Loads a .gltf/.glb file. Throws std::runtime_error on failure.
Mesh load_gltf(const std::string &path);
