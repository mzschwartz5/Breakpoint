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
        0.25, 0.25, 3.25,
        0.25, 0.5, 3.25,
        0.5, 0.5, 3.25,
        0.25, -0.75, 6.25,
        0.25, -0.5, 6.25,
        0.5, -0.5, 6.25,
    };
    unsigned int idxdata[] = {
        0, 1, 2,
        3, 4, 5
    };
    //number of bytes * number of vertices * number of floats per vertex
    VertexBuffer vertBuffer = VertexBuffer(vdata, 4 * 6 * 3, 12);
    auto vbv = vertBuffer.passVertexDataToGPU(context, cmdList);

    IndexBuffer idxBuffer = IndexBuffer(idxdata, 4 * 6);
    std::cout << sizeof(unsigned int);
    auto ibv = idxBuffer.passIndexDataToGPU(context, cmdList);

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
        //cmdList->DrawInstanced(3, 2, 0, 0);
        cmdList->DrawIndexedInstanced(3, 2, 0, 0, 0);

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