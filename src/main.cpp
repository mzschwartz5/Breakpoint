#include "main.h"

int main() {
    DebugLayer debugLayer = DebugLayer();
    DXContext context = DXContext();
    auto* cmdList = context.initCommandList();
    if (!Window::get().init(&context, SCREEN_WIDTH, SCREEN_HEIGHT)) {
        //handle could not initialize window
        std::cout << "could not initialize window\n";
        Window::get().shutdown();
        return false;
    }

    //pass memory to gpu, get vertex buffer view
    float vdata[] = {-1.f, -1.f,
      0.f,  1.f ,
      1.f, -1.f  };
    VertexBuffer buffer = VertexBuffer(vdata, 24, 8);
    auto vbv = buffer.passVertexDataToGPU(context, cmdList);

    RenderPipeline basicPipeline( "VertexShader.cso" , "PixelShader.cso" , "RootSignature.cso" , Standard2D, context);

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
        }

        //begin draw
        cmdList = context.initCommandList();

        //draw to window
        Window::get().beginFrame(cmdList);

        //draw
        // == IA ==
        cmdList->IASetVertexBuffers(0, 1, &vbv);
        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // == RS ==
        D3D12_VIEWPORT vp;
        createDefaultViewport(vp, cmdList);
        // == PSO ==
        cmdList->SetPipelineState(pso);
        cmdList->SetGraphicsRootSignature(rootSignature);
        // == ROOT ==
        static float color[] = { 0.0f, 0.0f, 0.0f };
        emitColor(color);
        cmdList->SetGraphicsRoot32BitConstants(0, 3, color, 0);

        // Draw
        cmdList->DrawInstanced(3, 1, 0, 0);

        Window::get().endFrame(cmdList);

        //finish draw, present
        context.executeCommandList();
        Window::get().present();
    }

    // Close
    buffer.releaseResources();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}