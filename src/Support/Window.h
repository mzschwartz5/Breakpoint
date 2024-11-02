#pragma once

#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"

class Window {
public:
	Window(const Window&) = delete;
	Window& operator=(const Window&) = delete;

	inline static Window& Get() {
		static Window instance;
		return instance;
	}

	inline bool ShouldClose() const {
		return m_shouldClose;
	}

	bool Init();
	void Update();
	void Shutdown();

	bool m_shouldClose = false;

private:
	Window() = default;

	static LRESULT CALLBACK OnWindowMessage(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
	ATOM m_wndClass = 0;
	HWND m_window = nullptr;

#ifdef _DEBUG
	ComPointer<ID3D12Debug3> m_d3d12Debug;
	ComPointer<IDXGIDebug1> m_dxgiDebug;
#endif
};