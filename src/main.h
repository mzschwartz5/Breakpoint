#include <iostream>

#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "Support/Window.h"
#include "Support/Shader.h"
#include "Debug/DebugLayer.h"
#include "D3D/DXContext.h"
#include "D3D/RenderPipelineHelper.h"
#include "D3D/RenderPipeline.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

void emitColor(float* color)
{
    static int pukeState = 0;
    color[pukeState] += 0.01f;
    if (color[pukeState] > 1.0f)
    {
        pukeState++;
        if (pukeState == 3)
        {
            color[0] = 0.0f;
            color[1] = 0.0f;
            color[2] = 0.0f;
            pukeState = 0;
        }
    }
}


