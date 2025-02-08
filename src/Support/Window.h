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

	D3D12_GPU_DESCRIPTOR_HANDLE getObjectColorTextureHandle() {
		return objectSceneSRVHandleColor;
	}

	D3D12_GPU_DESCRIPTOR_HANDLE getObjectPositionTextureHandle() {
		return objectSceneSRVHandlePosition;
	}

	bool init(DXContext* context, int w, int h);
	void update();

	void present();
	void resize();

	bool createObjectSceneRenderTargets(DescriptorHeap* srvDescHeap);
	void beginFrame(ID3D12GraphicsCommandList6* cmdList);
	void setMainRT(ID3D12GraphicsCommandList6* cmdList);
	void setTextureRTs(ID3D12GraphicsCommandList6* cmdList);
	void transitionSwapChain(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
	void transitionObjectRTs(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);

	void shutdown();

	void updateTitle(std::wstring text);

	static void createViewport(D3D12_VIEWPORT& vp);
	static void setViewport(D3D12_VIEWPORT& vp, ID3D12GraphicsCommandList5* cmdList);

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

	ComPointer<ID3D12DescriptorHeap> dsvDescHeap;
	D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
    ComPointer<ID3D12Resource> depthStencilBuffer;

	// Members for rendering object scene to frame buffer
	ComPointer<ID3D12Resource> objectSceneColorTexture;
	ComPointer<ID3D12Resource> objectScenePositionTexture;
	D3D12_CPU_DESCRIPTOR_HANDLE objectSceneRTVHandleColor;
	D3D12_CPU_DESCRIPTOR_HANDLE objectSceneRTVHandlePosition;
	D3D12_GPU_DESCRIPTOR_HANDLE objectSceneSRVHandleColor;
	D3D12_GPU_DESCRIPTOR_HANDLE objectSceneSRVHandlePosition;

	bool getBuffers();
	void releaseBuffers();

#ifdef _DEBUG
	ComPointer<ID3D12Debug3> d3d12Debug;
	ComPointer<IDXGIDebug1> dxgiDebug;
#endif
};