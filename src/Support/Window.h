#pragma once

#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"
#include "../D3D/DescriptorHeap.h"
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

	inline HWND& getHWND() {
		return window;
	}

	inline bool getShouldResize() const {
		return shouldResize;
	}

	inline bool getShouldClose() const {
		return shouldClose;
	}

	inline UINT getWidth() const {
		return width;
	}

	inline UINT getHeight() const {
		return height;
	}

	bool init(DXContext* context, int w, int h);
	void update();

	void present();
	void resize();

	void beginFrame(ID3D12GraphicsCommandList6* cmdList);
	void endFrame(ID3D12GraphicsCommandList6* cmdList);

	void shutdown();

	void renderText(ID3D12GraphicsCommandList6* cmdList, std::wstring text);

	static void createAndSetDefaultViewport(D3D12_VIEWPORT& vp, ID3D12GraphicsCommandList5* cmdList);
	static void getViewport(D3D12_VIEWPORT& vp);

private:
	Window() = default;

	DXContext* dxContext;

	UINT width, height;

	bool shouldResize = false;

	bool shouldClose = false;

	static LRESULT CALLBACK OnWindowMessage(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
	ATOM wndClass = 0;
	HWND window = nullptr;

	ComPointer<IDXGISwapChain4> swapChain;
	ComPointer<ID3D12Resource1> swapChainBuffers[FRAME_COUNT];
	size_t currentSwapChainBufferIdx = 0;

	ComPointer<ID3D12DescriptorHeap> rtvDescHeap;
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandles[FRAME_COUNT];

	bool getBuffers();
	void releaseBuffers();

	std::unique_ptr<DescriptorHeap> fontDH;
	std::unique_ptr<DirectX::SpriteFont> font;
	std::unique_ptr<DirectX::SpriteBatch> spriteBatch;
	DirectX::XMVECTOR fontPos;

#ifdef _DEBUG
	ComPointer<ID3D12Debug3> d3d12Debug;
	ComPointer<IDXGIDebug1> dxgiDebug;
#endif
};