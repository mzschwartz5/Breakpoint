#include "main.h"

// Base Object Scene = 0, Bouncing Ball Scene = 1, Mesh Shader Scene = 2, PBMPM Scene = 3
#define SCENE 0

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

    context.createCommandList(CommandListID::PAPA_ID);
    context.resetCommandList(CommandListID::PAPA_ID);

#if SCENE == 0
	RenderPipeline basicPipeline("InstancedVertexShader.cso", "PixelShader.cso", "RootSignature.cso", context, CommandListID::RENDER_ID,
	D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

	// Initialize command lists
	context.resetCommandList(CommandListID::RENDER_ID);

    ObjectScene scene{ &context, &basicPipeline };
#endif
#if SCENE == 1
    RenderPipeline basicPipeline("PhysicsVertexShader.cso", "PixelShader.cso", "PhysicsRootSignature.cso", context, CommandListID::RENDER_ID,
	D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

    // Create compute pipeline
    ComputePipeline computePipeline("TestComputeRootSignature.cso", "TestComputeShader.cso", context, CommandListID::PBMPM_COMPUTE_ID,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

	// Initialize command lists
    context.resetCommandList(CommandListID::RENDER_ID);
    context.resetCommandList(CommandListID::PBMPM_COMPUTE_ID);

    PhysicsScene scene{ &context, &basicPipeline, &computePipeline, 10 };
#endif
#if SCENE == 2
    MeshPipeline basicPipeline("MeshShader.cso", "PixelShader.cso", "RootSignature.cso", context,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    // TODO: Make a Scene Class for Mesh Shading?
#endif
#if SCENE == 3
    RenderPipeline basicPipeline("PBMPMVertexShader.cso", "PixelShader.cso", "PBMPMVertexRootSignature.cso", context, CommandListID::RENDER_ID,
	D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

    // Create compute pipeline
    ComputePipeline computePipeline("TestComputeRootSignature.cso", "TestComputeShader.cso", context, CommandListID::PBMPM_COMPUTE_ID,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

    // Initialize command lists
    context.resetCommandList(CommandListID::RENDER_ID);
    context.resetCommandList(CommandListID::PBMPM_COMPUTE_ID);

    PBMPMScene scene{ &context, &basicPipeline, &computePipeline, 10 };
#endif

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

        if (mState.positionMode == Mouse::MODE_RELATIVE) {
            camera->rotateOnX(-mState.y * 0.01f);
            camera->rotateOnY(mState.x * 0.01f);
            camera->rotate();
        }

        mouse->SetMode(mState.leftButton ? Mouse::MODE_RELATIVE : Mouse::MODE_ABSOLUTE);

        //update camera
        camera->updateViewMat();
#if SCENE == 1 || SCENE == 3
		// Dispatch compute shader for physics scene
        scene.compute();
#endif
        //draw to window
        Window::get().beginFrame(basicPipeline.getCommandList());
        D3D12_VIEWPORT vp;
        createDefaultViewport(vp, basicPipeline.getCommandList());

        scene.draw(camera.get());
        
        Window::get().endFrame(basicPipeline.getCommandList());

        //finish draw, present, reset
        context.executeCommandList(basicPipeline.getCommandListID());
        Window::get().present();
		context.resetCommandList(basicPipeline.getCommandListID());
    }

    // Close
    // Scene should release all resources, including their pipelines
    scene.releaseResources();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}