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
    context.getDevice()->CreateCommittedResource(&hpUpload, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ , nullptr, IID_PPV_ARGS(&uploadBuffer));
    context.getDevice()->CreateCommittedResource(&hpDefault, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&vertexBuffer));
    // Copy void* --> CPU Resource
    void* uploadBufferAddress;
    D3D12_RANGE uploadRange;
    uploadRange.Begin = 0;
    uploadRange.End = 1023;
    uploadBuffer->Map(0, &uploadRange, &uploadBufferAddress);
    memcpy(uploadBufferAddress, vertices, sizeof(vertices));
    uploadBuffer->Unmap(0, &uploadRange);
    // Copy CPU Resource --> GPU Resource
    auto* cmdList = context.initCommandList();
    cmdList->CopyBufferRegion(vertexBuffer, 0, uploadBuffer, 0, 1024);
    context.executeCommandList();

    // === Shaders ===
    Shader rootSignatureShader("root_signature.cso");
    Shader vertexShader("basic_2d_vert.cso");
    Shader pixelShader("basic_2d_frag.cso");

    // === Create root signature ===
    ComPointer<ID3D12RootSignature> rootSignature;
    context.getDevice()->CreateRootSignature(0, rootSignatureShader.getBuffer(), rootSignatureShader.getSize(), IID_PPV_ARGS(&rootSignature));

    // === Pipeline state ===
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gfxPsod{};
    gfxPsod.pRootSignature = rootSignature;
    gfxPsod.InputLayout.NumElements = _countof(vertexLayout);
    gfxPsod.InputLayout.pInputElementDescs = vertexLayout;
    gfxPsod.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;

    gfxPsod.VS.BytecodeLength = vertexShader.getSize();
    gfxPsod.VS.pShaderBytecode = vertexShader.getBuffer();

    gfxPsod.PS.BytecodeLength = pixelShader.getSize();
    gfxPsod.PS.pShaderBytecode = pixelShader.getBuffer();

    gfxPsod.DS.BytecodeLength = 0;
    gfxPsod.DS.pShaderBytecode = nullptr;
    gfxPsod.HS.BytecodeLength = 0;
    gfxPsod.HS.pShaderBytecode = nullptr;
    gfxPsod.GS.BytecodeLength = 0;
    gfxPsod.GS.pShaderBytecode = nullptr;
    gfxPsod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gfxPsod.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    gfxPsod.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gfxPsod.RasterizerState.FrontCounterClockwise = FALSE;
    gfxPsod.RasterizerState.DepthBias = 0;
    gfxPsod.RasterizerState.DepthBiasClamp = .0f;
    gfxPsod.RasterizerState.SlopeScaledDepthBias = .0f;
    gfxPsod.RasterizerState.DepthClipEnable = FALSE;
    gfxPsod.RasterizerState.MultisampleEnable = FALSE;
    gfxPsod.RasterizerState.AntialiasedLineEnable = FALSE;
    gfxPsod.RasterizerState.ForcedSampleCount = 0;
    gfxPsod.StreamOutput.NumEntries = 0;
    gfxPsod.StreamOutput.NumStrides = 0;
    gfxPsod.StreamOutput.pBufferStrides = nullptr;
    gfxPsod.StreamOutput.pSODeclaration = nullptr;
    gfxPsod.StreamOutput.RasterizedStream = 0;

    gfxPsod.NumRenderTargets = 1;
    gfxPsod.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    gfxPsod.DSVFormat = DXGI_FORMAT_UNKNOWN;
    gfxPsod.BlendState.AlphaToCoverageEnable = FALSE;
    gfxPsod.BlendState.IndependentBlendEnable = FALSE;
    gfxPsod.BlendState.RenderTarget[0].BlendEnable = FALSE;
    gfxPsod.BlendState.RenderTarget[0].LogicOpEnable = FALSE;
    gfxPsod.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ZERO;
    gfxPsod.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ZERO;
    gfxPsod.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    gfxPsod.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ZERO;
    gfxPsod.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    gfxPsod.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    gfxPsod.BlendState.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_NOOP;
    gfxPsod.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    gfxPsod.DepthStencilState.DepthEnable = FALSE;
    gfxPsod.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    gfxPsod.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    gfxPsod.DepthStencilState.StencilEnable = FALSE;
    gfxPsod.DepthStencilState.StencilReadMask = 0;
    gfxPsod.DepthStencilState.StencilWriteMask = 0;
    gfxPsod.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    gfxPsod.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.BackFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    gfxPsod.DepthStencilState.BackFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.BackFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.BackFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.SampleMask = 0xFFFFFFFF;
    gfxPsod.SampleDesc.Count = 1;
    gfxPsod.SampleDesc.Quality = 0;

    gfxPsod.NodeMask = 0;
    gfxPsod.CachedPSO.CachedBlobSizeInBytes = 0;
    gfxPsod.CachedPSO.pCachedBlob = nullptr;
    gfxPsod.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
 
    //output merger
    ComPointer<ID3D12PipelineState> pso;
    context.getDevice()->CreateGraphicsPipelineState(&gfxPsod, IID_PPV_ARGS(&pso));

    // === Vertex buffer view ===
    D3D12_VERTEX_BUFFER_VIEW vbv{};
    vbv.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
    vbv.SizeInBytes = sizeof(Vertex) * _countof(vertices);
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
        // == PSO ==
        cmdList->SetPipelineState(pso);
        cmdList->SetGraphicsRootSignature(rootSignature);
        // Draw
        cmdList->DrawInstanced(_countof(vertices), 1, 0, 0);

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