#pragma once

#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"

class DebugLayer {
public:

	DebugLayer();
	~DebugLayer();

#ifdef _DEBUG
	ComPointer<ID3D12Debug3> m_d3d12Debug;
	ComPointer<IDXGIDebug1> m_dxgiDebug;
#endif

	bool isInitialized();

private:

	bool initialized = false;
};