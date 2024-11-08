#pragma once

#include "Support/WinInclude.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

using namespace DirectX;

class Camera {
public:
	Camera();

	void setFOV(float FOVY, float aspect, float nearPlane, float farPlane);
	void updateAspect(float aspect);
	
	void rotateOnX(float angle);
	void rotateOnY(float angle);
	void rotate();
	void translate(XMFLOAT3 distance);

	void updateViewMat();

	XMMATRIX getViewMat();
	XMMATRIX getProjMat();

private:
	float FOVY;
	float aspect;
	float nearPlane;
	float farPlane;

	float rotateX{ 0 };
	float rotateY{ 0 };

	XMFLOAT3 position{ 0, 0, -1 };
	XMFLOAT3 up{ 0, 1, 0 };
	XMFLOAT3 forward{ 0, 0, 1 };
	XMFLOAT3 right{ 1, 0, 0 };

	XMFLOAT4X4 viewMat;
	XMFLOAT4X4 projMat;

	void updateProjMat();
};