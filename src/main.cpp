#include "main.h"
#include "Scene/Geometry.h"
#include "./PBD/particles.h"

struct Constants {
    float gravityStrength;
    float inputX;
    float inputY;
    float deltaTime;
};

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

    int instanceCount = 10;

    // Create Model Matrix
    auto modelMat = XMMatrixIdentity();
    modelMat *= XMMatrixTranslation(0.0f, 0.0f, 0.0f);

    // Create test position data
    std::vector<XMFLOAT3> positions;
    for (int i = 0; i < instanceCount; ++i) {
        positions.push_back({ -0.72f + 0.15f * i, 0.f, 0.f });
    }

    // Create test velocity data
    std::vector<XMFLOAT3> velocities;
    for (int i = 0; i < instanceCount; ++i) {
        velocities.push_back({ 0.0f, 0.0f, 0.0f });
    }

    std::vector<Particle> particles(instanceCount);
    for (int i = 0; i < instanceCount; ++i) {
        particles[i].position = positions[i];
        particles[i].prevPosition = particles[i].position;
        particles[i].velocity = { 0.0f, 0.0f, 0.0f };
        particles[i].invMass = 1.0f; // Assuming uniform mass for simplicity
    }

    // Create DistanceConstraint data
    std::vector<DistanceConstraint> constraints;
    for (int i = 0; i < instanceCount - 1; ++i) {
        DistanceConstraint dc;
        dc.particleA = i;
        dc.particleB = i + 1;
        dc.restLength =2.0f; // Distance between connected particles
        constraints.push_back(dc);
    }


    // Create buffer for particle data
    StructuredBuffer particleBuffer = StructuredBuffer(particles.data(), instanceCount, sizeof(Particle));

    // Create buffer for constraints
    StructuredBuffer constraintBuffer = StructuredBuffer(constraints.data(), constraints.size(), sizeof(DistanceConstraint));

    // Create compute pipeline
    ComputePipeline computePipeline("TestComputeRootSignature.cso", "TestComputeShader.cso", context, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);


    // Pass particle data to GPU as UAV
    particleBuffer.passUAVDataToGPU(context, computePipeline.getDescriptorHeap()->GetCPUHandleAt(0), cmdList);

    // Pass constraint data to GPU as SRV
    constraintBuffer.passSRVDataToGPU(context, computePipeline.getDescriptorHeap()->GetCPUHandleAt(1), cmdList);


    // Create circle geometry
    auto circleData = generateCircle(0.05f, 32);

    VertexBuffer vertBuffer = VertexBuffer(circleData.first, circleData.first.size() * sizeof(XMFLOAT3), sizeof(XMFLOAT3));
    auto vbv = vertBuffer.passVertexDataToGPU(context, cmdList);

    IndexBuffer idxBuffer = IndexBuffer(circleData.second, circleData.second.size() * sizeof(unsigned int));
    auto ibv = idxBuffer.passIndexDataToGPU(context, cmdList);

    //Transition both buffers to their usable states
    D3D12_RESOURCE_BARRIER barriers[2] = {};

    // Vertex buffer barrier
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = vertBuffer.getVertexBuffer().Get();
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    // Index buffer barrier
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = idxBuffer.getIndexBuffer().Get();
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmdList->ResourceBarrier(2, barriers);

    RenderPipeline basicPipeline("PhysicsVertexShader.cso", "PixelShader.cso", "PhysicsRootSignature.cso", context,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

    /*MeshPipeline basicPipeline("MeshShader.cso", "PixelShader.cso", "RootSignature.cso", context,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);*/

        // === Pipeline state ===
    basicPipeline.createPSOD();

    //output merger
    basicPipeline.createPipelineState(context.getDevice());

    context.executeCommandList();

    // Create a fence
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

        if (mState.positionMode == Mouse::MODE_RELATIVE) {
            camera->rotateOnX(-mState.y * 0.01);
            camera->rotateOnY(mState.x * 0.01);
            camera->rotate();
        }

        mouse->SetMode(mState.leftButton ? Mouse::MODE_RELATIVE : Mouse::MODE_ABSOLUTE);

        //update camera
        camera->updateViewMat();

        cmdList = context.initCommandList();
        // Transition position buffer to UAV for compute pass
        D3D12_RESOURCE_BARRIER UAVbarrier = {};
        UAVbarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        UAVbarrier.Transition.pResource = particleBuffer.getBuffer();
        UAVbarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        UAVbarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        UAVbarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList->ResourceBarrier(1, &UAVbarrier);

        // Set compute pipeline state and root signature
       
        cmdList->SetPipelineState(computePipeline.getPSO());
        cmdList->SetComputeRootSignature(computePipeline.getRootSignature());



        // Set descriptor heap and compute root descriptor table
        ID3D12DescriptorHeap* computeDescriptorHeaps[] = { computePipeline.getDescriptorHeap()->Get() };
        cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

        // Set compute root descriptor table for particles UAV
        cmdList->SetComputeRootDescriptorTable(0, computePipeline.getDescriptorHeap()->GetGPUHandleAt(0)); // Particles UAV

        // Set compute root descriptor table for constraints SRV
        cmdList->SetComputeRootDescriptorTable(1, computePipeline.getDescriptorHeap()->GetGPUHandleAt(1)); // Constraints SRV

        // Dispatch the compute shader to update the particle positions
        cmdList->Dispatch(instanceCount, 1, 1);

        // Execute the command list
        context.executeCommandList();
        context.signalAndWaitForFence(fence, fenceValue);

        //begin draw
        cmdList = context.initCommandList();

        // Transition position buffer back to SRV for vertex shader use
        D3D12_RESOURCE_BARRIER SRVbarrier = {};
        SRVbarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        SRVbarrier.Transition.pResource = particleBuffer.getBuffer();
        SRVbarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        SRVbarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        SRVbarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList->ResourceBarrier(1, &SRVbarrier);


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
        cmdList->SetPipelineState(basicPipeline.getPSO());
        cmdList->SetGraphicsRootSignature(basicPipeline.getRootSignature());
        // == ROOT ==

        ID3D12DescriptorHeap* descriptorHeaps[] = { basicPipeline.getDescriptorHeap()->Get() };
        cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

        cmdList->SetGraphicsRootDescriptorTable(1, particleBuffer.getSRVGPUHandle());

        auto viewMat = camera->getViewMat();
        auto projMat = camera->getProjMat();
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &modelMat, 32);

        // Draw
        cmdList->DrawIndexedInstanced(circleData.second.size(), instanceCount, 0, 0, 0);
        //cmdList->DispatchMesh(1, 1, 1);

        Window::get().endFrame(cmdList);

        //finish draw, present
        context.executeCommandList();
        Window::get().present();
    }

    // Close
    vertBuffer.releaseResources();
    idxBuffer.releaseResources();
    //modelBuffer.releaseResources();
    basicPipeline.releaseResources();
    particleBuffer.releaseResources();
    computePipeline.releaseResources();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}
