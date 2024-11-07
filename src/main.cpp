#include "main.h"

int main() {
    DebugLayer debugLayer = DebugLayer();
    DXContext context = DXContext();
    auto* cmdList = context.initCommandList();
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

    //pass triangle data to gpu, get vertex buffer view
    unsigned int idxdata[] = {
        0, 1, 2,
        3, 4, 5
    };

    float vdata[] = {
        0.25, 0.0, 1.25,
        0.25, 0.25, 1.25,
        0.5, 0.25, 1.25,
        0.25, -0.75, 6.25,
        0.25, -0.5, 6.25,
        0.5, -0.5, 6.25,
    };

    IndexBuffer idxBuffer = IndexBuffer(idxdata, sizeof(idxdata));
    auto ibv = idxBuffer.passIndexDataToGPU(context, cmdList);

    VertexBuffer vertBuffer = VertexBuffer(vdata, sizeof(vdata), 3 * sizeof(float));
    auto vbv = vertBuffer.passVertexDataToGPU(context, cmdList);

	// Execute command list
	context.executeCommandList();

    RenderPipeline basicPipeline("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", context);

    // === Create root signature ===
    ComPointer<ID3D12RootSignature> rootSignature = basicPipeline.getRootSignature();

    // === Pipeline state ===
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gfxPsod{};
    createShaderPSOD(gfxPsod, rootSignature, basicPipeline.getVertexShader(), basicPipeline.getFragmentShader());
    
    //output merger
    ComPointer<ID3D12PipelineState> pso;
    context.getDevice()->CreateGraphicsPipelineState(&gfxPsod, IID_PPV_ARGS(&pso));

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
            camera->translate({ 0, 0, 0.0005 });
        }
        if (kState.A) {
            camera->translate({ -0.0005, 0, 0 });
        }
        if (kState.S) {
            camera->translate({ 0, 0, -0.0005 });
        }
        if (kState.D) {
            camera->translate({ 0.0005, 0, 0 });
        }
        if (kState.Space) {
            camera->translate({ 0, 0.0005, 0 });
        }
        if (kState.LeftControl) {
            camera->translate({ 0, -0.0005, 0 });
        }

        //check mouse state
        auto mState = mouse->GetState();

        //update camera
        camera->updateViewMat();

        //begin draw
        cmdList = context.initCommandList();

        //draw to window
        Window::get().beginFrame(cmdList);

        //draw
        // == IA ==
        cmdList->IASetVertexBuffers(0, 1, &vbv);
		cmdList->IASetIndexBuffer(&ibv);
        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // == RS ==
        D3D12_VIEWPORT vp;
        createDefaultViewport(vp, cmdList);
        // == PSO ==
        cmdList->SetPipelineState(pso);
        cmdList->SetGraphicsRootSignature(rootSignature);
        // == ROOT ==
        auto viewMat = camera->getViewMat();
        auto projMat = camera->getProjMat();
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);

        // Draw
        cmdList->DrawIndexedInstanced(6, 1, 0, 0, 0);

        Window::get().endFrame(cmdList);

        //finish draw, present
        context.executeCommandList();
        Window::get().present();
    }

    // Close
    vertBuffer.releaseResources();
    idxBuffer.releaseResources();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}