#pragma once

#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"

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

	bool init();
	void update();
	void shutdown();

private:
	Window() = default;

	bool shouldClose = false;

	static LRESULT CALLBACK OnWindowMessage(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
	ATOM wndClass = 0;
	HWND window = nullptr;

#ifdef _DEBUG
	ComPointer<ID3D12Debug3> d3d12Debug;
	ComPointer<IDXGIDebug1> dxgiDebug;
#endif
};