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
        //update window
        Window::get().update();
        if (Window::get().getShouldResize()) {
            //flush pending buffer operations in swapchain
            context.flush(FRAME_COUNT);
            Window::get().resize();
        }

        //begin draw
        auto* cmdList = context.initCommandList();

        //finish draw, present
        context.executeCommandList();
        Window::get().present();
    }

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}