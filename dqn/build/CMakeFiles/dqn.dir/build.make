# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rantd/dqn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rantd/dqn/build

# Include any dependencies generated for this target.
include CMakeFiles/dqn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dqn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dqn.dir/flags.make

CMakeFiles/dqn.dir/main.cpp.o: CMakeFiles/dqn.dir/flags.make
CMakeFiles/dqn.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rantd/dqn/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/dqn.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/dqn.dir/main.cpp.o -c /home/rantd/dqn/main.cpp

CMakeFiles/dqn.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dqn.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rantd/dqn/main.cpp > CMakeFiles/dqn.dir/main.cpp.i

CMakeFiles/dqn.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dqn.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rantd/dqn/main.cpp -o CMakeFiles/dqn.dir/main.cpp.s

CMakeFiles/dqn.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/dqn.dir/main.cpp.o.requires

CMakeFiles/dqn.dir/main.cpp.o.provides: CMakeFiles/dqn.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/dqn.dir/build.make CMakeFiles/dqn.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/dqn.dir/main.cpp.o.provides

CMakeFiles/dqn.dir/main.cpp.o.provides.build: CMakeFiles/dqn.dir/main.cpp.o

# Object files for target dqn
dqn_OBJECTS = \
"CMakeFiles/dqn.dir/main.cpp.o"

# External object files for target dqn
dqn_EXTERNAL_OBJECTS =

dqn: CMakeFiles/dqn.dir/main.cpp.o
dqn: CMakeFiles/dqn.dir/build.make
dqn: /usr/lib/x86_64-linux-gnu/libboost_system.so
dqn: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
dqn: /usr/lib/x86_64-linux-gnu/libSDLmain.a
dqn: /usr/lib/x86_64-linux-gnu/libSDL.so
dqn: /usr/lib/x86_64-linux-gnu/libSDL_image.so
dqn: CMakeFiles/dqn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable dqn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dqn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dqn.dir/build: dqn
.PHONY : CMakeFiles/dqn.dir/build

CMakeFiles/dqn.dir/requires: CMakeFiles/dqn.dir/main.cpp.o.requires
.PHONY : CMakeFiles/dqn.dir/requires

CMakeFiles/dqn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dqn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dqn.dir/clean

CMakeFiles/dqn.dir/depend:
	cd /home/rantd/dqn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rantd/dqn /home/rantd/dqn /home/rantd/dqn/build /home/rantd/dqn/build /home/rantd/dqn/build/CMakeFiles/dqn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dqn.dir/depend

