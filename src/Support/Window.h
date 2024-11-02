#pragma once

#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"
#include "../D3D/DXContext.h"

#define FRAME_COUNT 2

class Window {
public:
	Window(const Window&) = delete;
	Window& operator=(const Window&) = delete;

	inline static Window& get() {
		static Window instance;
		return instance;
	}

	inline bool getShouldClose() const {
		return shouldClose;
	}

	bool init(DXContext* context, int w, int h);
	void update();
	void present();
	void shutdown();

private:
	Window() = default;

	int width{ 1920 }, height{ 1080 };

	bool shouldClose = false;

	static LRESULT CALLBACK OnWindowMessage(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
	ATOM wndClass = 0;
	HWND window = nullptr;

	ComPointer<IDXGISwapChain4> swapChain;

	DXContext* dxContext;

#ifdef _DEBUG
	ComPointer<ID3D12Debug3> d3d12Debug;
	ComPointer<IDXGIDebug1> dxgiDebug;
#endif
};