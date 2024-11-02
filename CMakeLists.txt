cmake_minimum_required(VERSION 3.18)
project(breakpoint LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

# Add directories for Direct3D 12
set(D3D12_INCLUDE_DIR "C:/Program Files (x86)/Windows Kits/10/Include/10.0.18362.0/um")
set(D3D12_LIB_DIR "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.18362.0/um/x64")

include_directories(${D3D12_INCLUDE_DIR})
link_directories(${D3D12_LIB_DIR})

# Add D3D12 libraries
set(D3D12_LIBS d3d12 dxgi d3dcompiler)

set(headers
    src/main.h
)

set(sources
    src/main.cpp
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

set (support_headers     
     src/Support/WinInclude.h
     src/Support/ComPointer.h
     src/Support/Window.h
     src/Support/pch.h
)

set (support_sources     
     src/Support/Window.cpp
     src/Support/pch.cpp
)

set (debuglayer_headers
     src/DebugLayer/DebugLayer.h
)

set (debuglayer_sources
     src/DebugLayer/DebugLayer.cpp
)

set (d3d_sources
     src/D3D/DXContext.cpp
)

set (d3d_headers
     src/D3D/StepTimer.h
     src/D3D/DirectXHelper.h
     src/D3D/d3dx12.h
     src/D3D/DXContext.h
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

list(SORT support_sources)
list(SORT support_headers)
list(SORT debuglayer_sources)
list(SORT debuglayer_headers)
list(SORT d3d_headers)
list(SORT d3d_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

source_group("Support\\Headers" FILES ${support_headers})
source_group("Support\\Sources" FILES ${support_sources})
source_group("DebugLayer\\Headers" FILES ${debuglayer_headers})
source_group("DebugLayer\\Sources" FILES ${debuglayer_sources})
source_group("D3D\\Headers" FILES ${d3d_headers})
source_group("D3D\\Sources" FILES ${d3d_sources})

#add_subdirectory(src/ImGui)
#add_subdirectory(src/Support)
#add_subdirectory(src/DebugLayer)
#add_subdirectory(src/D3D)

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers} ${support_sources} ${support_headers} ${debuglayer_sources} ${debuglayer_headers} ${d3d_sources} ${d3d_headers})

target_precompile_headers(${PROJECT_NAME} PRIVATE src/Support/pch.h)

target_link_libraries(${PROJECT_NAME} 
    d3d12 
    dxgi 
    d3dcompiler 
    ${D3D_LIBRARIES}
)

