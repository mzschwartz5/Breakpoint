#include "main.h"

// This should probably go somewhere else
void createDefaultViewport(D3D12_VIEWPORT& vp, ID3D12GraphicsCommandList5* cmdList) {
    vp.TopLeftX = vp.TopLeftY = 0;
    vp.Width = Window::get().getWidth();
    vp.Height = Window::get().getHeight();
    vp.MinDepth = 1.f;
    vp.MaxDepth = 0.f;
    cmdList->RSSetViewports(1, &vp);
    RECT scRect;
    scRect.left = scRect.top = 0;
    scRect.right = Window::get().getWidth();
    scRect.bottom = Window::get().getHeight();
    cmdList->RSSetScissorRects(1, &scRect);
}

int main() {
    //set up DX, window, keyboard mouse
    DebugLayer debugLayer = DebugLayer();
    DXContext context = DXContext();
    ID3D12GraphicsCommandList5* cmdList = context.initCommandList();
    std::unique_ptr<Camera> camera = std::make_unique<Camera>();
    std::unique_ptr<Keyboard> keyboard = std::make_unique<Keyboard>();
    std::unique_ptr<Mouse> mouse = std::make_unique<Mouse>();

    if (!Window::get().init(&context, SCREEN_WIDTH, SCREEN_HEIGHT)) {
        //handle could not initialize window
        std::cout << "could not initialize window\n";
        Window::get().shutdown();
        return false;
    }

    mouse->SetWindow(Window::get().getHWND());

    RenderPipeline basicPipeline("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", context,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

	/*MeshPipeline basicPipeline("MeshShader.cso", "PixelShader.cso", "RootSignature.cso", context,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);*/

    // === Pipeline state ===
    basicPipeline.createPSOD();

    //output merger
    basicPipeline.createPipelineState(context.getDevice());

    //set up scene
    Scene scene{ &context, &basicPipeline, cmdList };

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
            camera->translate({ 0.f, 0.f, 0.0005f });
        }
        if (kState.A) {
            camera->translate({ -0.0005f, 0.f, 0.f });
        }
        if (kState.S) {
            camera->translate({ 0.f, 0.f, -0.0005f });
        }
        if (kState.D) {
            camera->translate({ 0.0005f, 0.f, 0.f });
        }
        if (kState.Space) {
            camera->translate({ 0.f, 0.0005f, 0.f });
        }
        if (kState.LeftControl) {
            camera->translate({ 0.f, -0.0005f, 0.f });
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

        //begin draw
        cmdList = context.initCommandList();

        //draw to window
        Window::get().beginFrame(cmdList);

        D3D12_VIEWPORT vp;
        createDefaultViewport(vp, cmdList);

        //Draw scene
        scene.draw(basicPipeline.getPSO(), basicPipeline.getRootSignature(), camera.get());

        Window::get().endFrame(cmdList);

        //finish draw, present
        context.executeCommandList();
        Window::get().present();
    }

    // Close
	basicPipeline.releaseResources();
    scene.releaseResources();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}