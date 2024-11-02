#pragma once

#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"

class DebugLayer {
public:
	DebugLayer(const DebugLayer&) = delete;
	DebugLayer& operator=(const DebugLayer&) = delete;

	inline static DebugLayer& Get() {
		static DebugLayer instance;
		return instance;
	}

	bool Init();
	void Shutdown();

private:
	DebugLayer() = default;

#ifdef _DEBUG
	ComPointer<ID3D12Debug3> m_d3d12Debug;
	ComPointer<IDXGIDebug1> m_dxgiDebug;
#endif
};