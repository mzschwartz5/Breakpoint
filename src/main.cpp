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
    Scene scene{PBMPM, camera.get(), &context};

    PBMPMConstants pbmpmConstants{ {512, 512, 512}, 0.01F, 2.5F, 1.5F, 0.01F,
        (unsigned int)std::ceil(std::pow(10, 7)),
        1, 8, 30, 0, 0,  0, 0, 0, 0, 5, 0.9F };
    PBMPMConstants pbmpmTempConstants = pbmpmConstants;

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
        if (kState.D1) {
            scene.setRenderScene(Object);
        }
        if (kState.D2) {
            scene.setRenderScene(PBMPM);
        }
        if (kState.D3) {
            scene.setRenderScene(Fluid);
        }

        //check mouse state
        auto mState = mouse->GetState();

        mouse->SetMode(mState.leftButton ? Mouse::MODE_RELATIVE : Mouse::MODE_ABSOLUTE);

        if (mState.positionMode == Mouse::MODE_RELATIVE && kState.LeftShift) {
            camera->rotateOnX(-mState.y * 0.01f);
            camera->rotateOnY(mState.x * 0.01f);
            camera->rotate();
        }

        if (mState.rightButton) {
            //enable mouse force
            pbmpmTempConstants.mouseActivation = 1;

            POINT cursorPos;
            GetCursorPos(&cursorPos);

            float ndcX = (2.0f * cursorPos.x) / SCREEN_WIDTH - 1.0f;
            float ndcY = -(2.0f * cursorPos.y) / SCREEN_HEIGHT + 1.0f;

            XMVECTOR screenCursorPos = XMVectorSet(ndcX, ndcY, 0.0f, 1.0f);
            XMVECTOR worldCursorPos = XMVector4Transform(screenCursorPos, camera->getInvViewProjMat());
            XMStoreFloat4(&(pbmpmTempConstants.mousePosition), worldCursorPos);

            pbmpmTempConstants.mouseFunction = 0;
            pbmpmTempConstants.mouseRadius = 1000;
            scene.updatePBMPMConstants(pbmpmTempConstants);
        }
        else {
            pbmpmTempConstants.mouseActivation = 0;
        }

        //update camera
        camera->updateViewMat();

        //draw to window
        auto renderPipeline = scene.getRenderPipeline();
        auto meshPipeline = scene.getMeshPipeline();
        scene.compute();

        //begin frame
        Window::get().beginFrame(renderPipeline->getCommandList());

        //create viewport
        D3D12_VIEWPORT vp;
        Window::get().createViewport(vp, renderPipeline->getCommandList());

        //first render pass
        Window::get().setRT(renderPipeline->getCommandList());
        Window::get().setViewport(vp, renderPipeline->getCommandList());
        scene.draw();

        //set up ImGUI for frame
        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        //draw ImGUI
        drawImGUIWindow(pbmpmTempConstants, io);

        //render ImGUI
        ImGui::Render();
        if (!PBMPMScene::constantsEqual(pbmpmTempConstants, pbmpmConstants)) {
            scene.updatePBMPMConstants(pbmpmTempConstants);
            pbmpmConstants = pbmpmTempConstants;
        }

        meshPipeline->getCommandList()->SetDescriptorHeaps(1, &imguiSRVHeap);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), renderPipeline->getCommandList());

        //finish draw, present, reset
        context.executeCommandList(renderPipeline->getCommandListID());

        //mesh render pass
        Window::get().setRT(scene.getMeshPipeline()->getCommandList());
        Window::get().setViewport(vp, scene.getMeshPipeline()->getCommandList());
        scene.drawFluid();
        context.executeCommandList(scene.getMeshPipeline()->getCommandListID()); 

        //end frame
        Window::get().endFrame(scene.getMeshPipeline()->getCommandList());

        Window::get().present();
		context.resetCommandList(renderPipeline->getCommandListID());
        context.resetCommandList(scene.getMeshPipeline()->getCommandListID());
    }

    // Scene should release all resources, including their pipelines
    scene.releaseResources();

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    imguiSRVHeap->Release();

    //flush pending buffer operations in swapchain
    context.flush(FRAME_COUNT);
    Window::get().shutdown();
}
