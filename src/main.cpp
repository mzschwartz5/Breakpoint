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

    //pass memory to gpu
    D3D12_HEAP_PROPERTIES hpUpload{};
    hpUpload.Type = D3D12_HEAP_TYPE_UPLOAD;
    hpUpload.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hpUpload.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hpUpload.CreationNodeMask = 0;
    hpUpload.VisibleNodeMask = 0;
    D3D12_HEAP_PROPERTIES hpDefault{};
    hpDefault.Type = D3D12_HEAP_TYPE_DEFAULT;
    hpDefault.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hpDefault.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hpDefault.CreationNodeMask = 0;
    hpDefault.VisibleNodeMask = 0;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    rd.Width = 1024;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;
    ComPointer<ID3D12Resource1> uploadBuffer, vertexBuffer;
    context.getDevice()->CreateCommittedResource(&hpUpload, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&uploadBuffer));
    context.getDevice()->CreateCommittedResource(&hpDefault, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&vertexBuffer));
    // Copy void* --> CPU Resource
    void* uploadBufferAddress;
    D3D12_RANGE uploadRange;
    uploadRange.Begin = 0;
    uploadRange.End = 1023;
    uploadBuffer->Map(0, &uploadRange, &uploadBufferAddress);
    memcpy(uploadBufferAddress, verticies, sizeof(verticies));
    uploadBuffer->Unmap(0, &uploadRange);
    // Copy CPU Resource --> GPU Resource
    auto* cmdList = context.initCommandList();
    cmdList->CopyBufferRegion(vertexBuffer, 0, uploadBuffer, 0, 1024);
    context.executeCommandList();

    // === Shaders ===
    Shader vertexShader("basic_2d_vert.cso");
    Shader pixelShader("basic_2d_frag.cso");

    // === Pipeline state ===
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gfxPsod{};
    gfxPsod.InputLayout.NumElements = _countof(vertexLayout);
    gfxPsod.InputLayout.pInputElementDescs = vertexLayout;
    gfxPsod.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;

    gfxPsod.VS.BytecodeLength = vertexShader.getSize();
    gfxPsod.VS.pShaderBytecode = vertexShader.getBuffer();
    // TODO: Rasterizer
    gfxPsod.PS.BytecodeLength = vertexShader.getSize();
    gfxPsod.PS.pShaderBytecode = vertexShader.getBuffer();
    // TODO: OutputMerger

    // === Vertex buffer view ===
    D3D12_VERTEX_BUFFER_VIEW vbv{};
    vbv.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
    vbv.SizeInBytes = sizeof(Vertex) * _countof(verticies);
    vbv.StrideInBytes = sizeof(Vertex);

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
        // Draw
        cmdList->DrawInstanced(_countof(verticies), 1, 0, 0);

        Window::get().endFrame(cmdList);

        //finish draw, present
        context.executeCommandList();
        Window::get().present();
    }

    // Close
    vertexBuffer.Release();
    uploadBuffer.Release();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}