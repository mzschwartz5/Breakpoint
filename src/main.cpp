#include "main.h"

int main() {
    DebugLayer debugLayer = DebugLayer();
    DXContext context = DXContext();
    if (!Window::get().init(&context, SCREEN_WIDTH, SCREEN_HEIGHT)) {
        //handle could not initialize window
        std::cout << "could not initialize window\n";
        Window::get().shutdown();
        return false;
    }

    while (!Window::get().getShouldClose()) {
        Window::get().update();
        auto* cmdList = context.initCommandList();
        context.executeCommandList();
        Window::get().present();
    }

    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}