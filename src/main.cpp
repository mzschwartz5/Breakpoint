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

    //initialize ImGUI
    ImGuiIO& io = initImGUI(context);

    //set mouse to use the window
    mouse->SetWindow(Window::get().getHWND());

    //initialize scene
    Scene scene{camera.get(), &context};
    // Render target SRVs need to be in the scene SRV heap to be accessible by the shaders alongside the other resources (can only bind one SRV heap to a shader)
    Window::get().createObjectSceneRenderTargets(scene.getSRVHeap());

    UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;
    context.getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

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

        //check mouse state
        auto mState = mouse->GetState();

        mouse->SetMode(mState.leftButton ? Mouse::MODE_RELATIVE : Mouse::MODE_ABSOLUTE);

        if (mState.positionMode == Mouse::MODE_RELATIVE && kState.LeftShift) {
            camera->rotateOnX(-mState.y * 0.01f);
            camera->rotateOnY(mState.x * 0.01f);
            camera->rotate();
        }

        //update camera
        camera->updateViewMat();

        auto meshPipeline = scene.getMeshPipeline();
        auto objectPipeline = scene.getObjectPipeline();

        //create viewport
        D3D12_VIEWPORT vp;
        Window::get().createViewport(vp);

        // Object render pass
        Window::get().setViewport(vp, objectPipeline->getCommandList());
        Window::get().setObjectRTs(objectPipeline->getCommandList());
        scene.drawObjects();
        Window::get().transitionObjectRTs(objectPipeline->getCommandList(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        context.executeCommandList(objectPipeline->getCommandListID());
        context.signalAndWaitForFence(fence, fenceValue);
        context.resetCommandList(objectPipeline->getCommandListID());
        
        scene.compute();

        //mesh render pass (uses object color and position textures)
        Window::get().setViewport(vp, meshPipeline->getCommandList());
        Window::get().setFluidRT(meshPipeline->getCommandList());
        scene.drawFluid(
            Window::get().getObjectColorTextureHandle(),
            Window::get().getObjectPositionTextureHandle()
        );

        //end frame
        Window::get().transitionSwapChain(meshPipeline->getCommandList(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        context.executeCommandList(meshPipeline->getCommandListID());
        context.resetCommandList(meshPipeline->getCommandListID());

        Window::get().present();
    }

    // Scene should release all resources, including their pipelines
    scene.releaseResources();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}
