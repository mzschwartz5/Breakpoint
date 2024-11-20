#include "main.h"

int main() {
    //set up DX, window, keyboard mouse
    DebugLayer debugLayer = DebugLayer();
    DXContext context = DXContext();
    std::unique_ptr<Camera> camera = std::make_unique<Camera>();
    std::unique_ptr<Keyboard> keyboard = std::make_unique<Keyboard>();
    std::unique_ptr<Mouse> mouse = std::make_unique<Mouse>();

    if (!Window::get().init(&context, SCREEN_WIDTH, SCREEN_HEIGHT)) {
        //handle could not initialize window
        std::cout << "could not initialize window\n";
        Window::get().shutdown();
        return false;
    }

    //initialize FPS counter variables
    LARGE_INTEGER timeFrequency, startTime, endTime;
    float fps = 0.0f;
    int frameCount = 0;

    QueryPerformanceFrequency(&timeFrequency);
    QueryPerformanceCounter(&startTime);

    mouse->SetWindow(Window::get().getHWND());

    Scene scene{Object, camera.get(), &context};

    while (!Window::get().getShouldClose()) {
        //update window
        Window::get().update();
        if (Window::get().getShouldResize()) {
            //flush pending buffer operations in swapchain
            context.flush(FRAME_COUNT);
            Window::get().resize();
            camera->updateAspect((float)Window::get().getWidth() / (float)Window::get().getHeight());
        }
        
        //check keyboard state
        auto kState = keyboard->GetState();
        if (kState.W) {
            camera->translate({ 0.f, 0.f, 1.0f });
        }
        if (kState.A) {
            camera->translate({ -1.0f, 0.f, 0.f });
        }
        if (kState.S) {
            camera->translate({ 0.f, 0.f, -1.0f });
        }
        if (kState.D) {
            camera->translate({ 1.0f, 0.f, 0.f });
        }
        if (kState.Space) {
            camera->translate({ 0.f, 1.0f, 0.f });
        }
        if (kState.LeftControl) {
            camera->translate({ 0.f, -1.0f, 0.f });
        }
        if (kState.D1) {
            scene.setRenderScene(Object);
        }
        if (kState.D2) {
            scene.setRenderScene(PBMPM);
        }
        if (kState.D3) {
            scene.setRenderScene(Physics);
        }

        //check mouse state
        auto mState = mouse->GetState();

        if (mState.positionMode == Mouse::MODE_RELATIVE) {
            camera->rotateOnX(-mState.y * 0.01f);
            camera->rotateOnY(mState.x * 0.01f);
            camera->rotate();
        }

        mouse->SetMode(mState.leftButton ? Mouse::MODE_RELATIVE : Mouse::MODE_ABSOLUTE);

        //update camera
        camera->updateViewMat();

        //draw to window
        auto renderPipeline = scene.getRenderPipeline();
        scene.compute();

        Window::get().beginFrame(renderPipeline->getCommandList());
        D3D12_VIEWPORT vp;
        Window::get().createAndSetDefaultViewport(vp, renderPipeline->getCommandList());

        scene.draw();

        //measure FPS
        QueryPerformanceCounter(&endTime);
        float elapsedTime = (float)(endTime.QuadPart - startTime.QuadPart) / (float)timeFrequency.QuadPart;
        frameCount++;

        if (elapsedTime >= 1.0f) {
            fps = (float)frameCount / elapsedTime;
            frameCount = 0;
            startTime = endTime;
        }

        std::wstringstream fpsStream;
        fpsStream << std::fixed << std::setprecision(2) << fps;
        std::wstring fpsStr = L"Breakpoint - FPS: " + fpsStream.str();
        Window::get().updateTitle(fpsStr);
        
        Window::get().endFrame(renderPipeline->getCommandList());

        //finish draw, present, reset
        context.executeCommandList(renderPipeline->getCommandListID());
        Window::get().present();
		context.resetCommandList(renderPipeline->getCommandListID());
    }

    // Close
    // Scene should release all resources, including their pipelines
    scene.releaseResources();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}