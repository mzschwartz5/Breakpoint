#include "Scene.h"

Scene::Scene(DXContext* p_context, RenderPipeline* p_pipeline) : context(p_context), pipeline(p_pipeline) {       
}

void Scene::releaseResources() {
	pipeline->releaseResources();
}
