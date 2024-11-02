#include <iostream>

#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "Support/Window.h"
#include "DebugLayer/DebugLayer.h"
#include "D3D/DXContext.h"

int main() {
    DebugLayer::Get().Init();

    if (DXContext::Get().Init() && Window::Get().Init()) {

        while (!Window::Get().ShouldClose()) {
            Window::Get().Update();
            auto* cmdList = DXContext::Get().InitCommandList();
            DXContext::Get().ExecuteCommandList();
        }
        Window::Get().Shutdown();
        DXContext::Get().Shutdown();
    }

    DebugLayer::Get().Shutdown();
}