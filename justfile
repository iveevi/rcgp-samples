rcgp_dir := justfile_directory() + "/ext/rcgp"

# Configure the samples build (builds rcgp as a subdirectory)
configure:
	cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug -DRCGP_DIR={{rcgp_dir}} .

# Build all samples
build: configure
	cmake --build build -j1

# Build a single sample, e.g. `just sample 01_triangle`
sample name: configure
	cmake --build build -t {{name}} -j1

# Remove the build directory
clean:
	rm -rf build
