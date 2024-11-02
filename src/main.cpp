#include "main.h"

int main() {
    DebugLayer debugLayer = DebugLayer();
    DXContext context = DXContext();
    if (!Window::get().init()) {
        return false;
    }

    while (!Window::get().getShouldClose()) {
        Window::get().update();
        auto* cmdList = context.initCommandList();
        context.executeCommandList();
    }
}