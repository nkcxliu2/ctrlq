# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.21.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.21.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/chenxuliu/Documents/GitHub/ctrlq

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/chenxuliu/Documents/GitHub/ctrlq/build

# Include any dependencies generated for this target.
include ctrlq/lib/CMakeFiles/solve.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include ctrlq/lib/CMakeFiles/solve.dir/compiler_depend.make

# Include the progress variables for this target.
include ctrlq/lib/CMakeFiles/solve.dir/progress.make

# Include the compile flags for this target's objects.
include ctrlq/lib/CMakeFiles/solve.dir/flags.make

ctrlq/lib/CMakeFiles/solve.dir/solve.cc.o: ctrlq/lib/CMakeFiles/solve.dir/flags.make
ctrlq/lib/CMakeFiles/solve.dir/solve.cc.o: ../ctrlq/lib/solve.cc
ctrlq/lib/CMakeFiles/solve.dir/solve.cc.o: ctrlq/lib/CMakeFiles/solve.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/chenxuliu/Documents/GitHub/ctrlq/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ctrlq/lib/CMakeFiles/solve.dir/solve.cc.o"
	cd /Users/chenxuliu/Documents/GitHub/ctrlq/build/ctrlq/lib && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ctrlq/lib/CMakeFiles/solve.dir/solve.cc.o -MF CMakeFiles/solve.dir/solve.cc.o.d -o CMakeFiles/solve.dir/solve.cc.o -c /Users/chenxuliu/Documents/GitHub/ctrlq/ctrlq/lib/solve.cc

ctrlq/lib/CMakeFiles/solve.dir/solve.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/solve.dir/solve.cc.i"
	cd /Users/chenxuliu/Documents/GitHub/ctrlq/build/ctrlq/lib && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/chenxuliu/Documents/GitHub/ctrlq/ctrlq/lib/solve.cc > CMakeFiles/solve.dir/solve.cc.i

ctrlq/lib/CMakeFiles/solve.dir/solve.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/solve.dir/solve.cc.s"
	cd /Users/chenxuliu/Documents/GitHub/ctrlq/build/ctrlq/lib && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/chenxuliu/Documents/GitHub/ctrlq/ctrlq/lib/solve.cc -o CMakeFiles/solve.dir/solve.cc.s

# Object files for target solve
solve_OBJECTS = \
"CMakeFiles/solve.dir/solve.cc.o"

# External object files for target solve
solve_EXTERNAL_OBJECTS =

../ctrlq/lib/solve.so: ctrlq/lib/CMakeFiles/solve.dir/solve.cc.o
../ctrlq/lib/solve.so: ctrlq/lib/CMakeFiles/solve.dir/build.make
../ctrlq/lib/solve.so: ctrlq/lib/libgetham.a
../ctrlq/lib/solve.so: ctrlq/lib/CMakeFiles/solve.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/chenxuliu/Documents/GitHub/ctrlq/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../../../ctrlq/lib/solve.so"
	cd /Users/chenxuliu/Documents/GitHub/ctrlq/build/ctrlq/lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/solve.dir/link.txt --verbose=$(VERBOSE)
	cd /Users/chenxuliu/Documents/GitHub/ctrlq/build/ctrlq/lib && /Library/Developer/CommandLineTools/usr/bin/strip -x /Users/chenxuliu/Documents/GitHub/ctrlq/ctrlq/lib/solve.so

# Rule to build all files generated by this target.
ctrlq/lib/CMakeFiles/solve.dir/build: ../ctrlq/lib/solve.so
.PHONY : ctrlq/lib/CMakeFiles/solve.dir/build

ctrlq/lib/CMakeFiles/solve.dir/clean:
	cd /Users/chenxuliu/Documents/GitHub/ctrlq/build/ctrlq/lib && $(CMAKE_COMMAND) -P CMakeFiles/solve.dir/cmake_clean.cmake
.PHONY : ctrlq/lib/CMakeFiles/solve.dir/clean

ctrlq/lib/CMakeFiles/solve.dir/depend:
	cd /Users/chenxuliu/Documents/GitHub/ctrlq/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/chenxuliu/Documents/GitHub/ctrlq /Users/chenxuliu/Documents/GitHub/ctrlq/ctrlq/lib /Users/chenxuliu/Documents/GitHub/ctrlq/build /Users/chenxuliu/Documents/GitHub/ctrlq/build/ctrlq/lib /Users/chenxuliu/Documents/GitHub/ctrlq/build/ctrlq/lib/CMakeFiles/solve.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ctrlq/lib/CMakeFiles/solve.dir/depend

