#pragma once

#define NOMINMAX

#include <Windows.h>

#include <wincodec.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <DirectXMath.h>
#include "./includes/d3dx12.h"

#include <Keyboard.h>
#include <Mouse.h>

#include "../ImGUI/imgui.h"
#include "../ImGUI/backends/imgui_impl_win32.h"
#include "../ImGUI/backends/imgui_impl_dx12.h"

#ifdef _DEBUG
#include <d3d12sdklayers.h>
#include <dxgidebug.h>
#endif