#!/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/bash

# Remove existing build directory
rm -rf build/

# Create a new build directory
mkdir build

# Move into the build directory
cd build || exit 1  # Exit if cd fails

# Run CMake to configure the project
cmake ..

# Compile the project
make

# Run the generated executable
./main