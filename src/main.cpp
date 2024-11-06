#include "main.h"

int main() {
    DebugLayer debugLayer = DebugLayer();
    DXContext context = DXContext();
    auto* cmdList = context.initCommandList();
    Camera camera = Camera();

    if (!Window::get().init(&context, SCREEN_WIDTH, SCREEN_HEIGHT)) {
        //handle could not initialize window
        std::cout << "could not initialize window\n";
        Window::get().shutdown();
        return false;
    }

    //pass memory to gpu, get vertex buffer view
    float vdata[] = {
        // Front face (z = 1)
        -1.f, -1.f,  1.f,
         1.f, -1.f,  1.f,
         1.f,  1.f,  1.f,
        -1.f, -1.f,  1.f,
         1.f,  1.f,  1.f,
        -1.f,  1.f,  1.f,

        // Back face (z = -1)
        -1.f, -1.f, -1.f,
        -1.f,  1.f, -1.f,
         1.f, -1.f, -1.f,
        -1.f,  1.f, -1.f,
         1.f,  1.f, -1.f,
         1.f, -1.f, -1.f,

         // Left face (x = -1)
         -1.f, -1.f, -1.f,
         -1.f, -1.f,  1.f,
         -1.f,  1.f,  1.f,
         -1.f, -1.f, -1.f,
         -1.f,  1.f,  1.f,
         -1.f,  1.f, -1.f,

         // Right face (x = 1)
          1.f, -1.f, -1.f,
          1.f,  1.f, -1.f,
          1.f, -1.f,  1.f,
          1.f,  1.f, -1.f,
          1.f,  1.f,  1.f,
          1.f, -1.f,  1.f,

          // Top face (y = 1)
          -1.f,  1.f, -1.f,
           1.f,  1.f, -1.f,
           1.f,  1.f,  1.f,
          -1.f,  1.f, -1.f,
           1.f,  1.f,  1.f,
          -1.f,  1.f,  1.f,

          // Bottom face (y = -1)
          -1.f, -1.f, -1.f,
           1.f, -1.f,  1.f,
           1.f, -1.f, -1.f,
          -1.f, -1.f, -1.f,
          -1.f, -1.f,  1.f,
           1.f, -1.f,  1.f
    };
    VertexBuffer buffer = VertexBuffer(vdata, 4 * 36, 12);
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
        camera.updateViewMat();
        auto viewMat = camera.getViewMat();
        auto projMat = camera.getProjMat();
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);

        // Draw
        cmdList->DrawInstanced(3, 12, 0, 0);

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