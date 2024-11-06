#pragma once

#include "Support/WinInclude.h"

using namespace DirectX;

class Camera {
public:
	Camera();

	void setFOV(float FOVY, float aspect, float nearPlane, float farPlane);
	void rotateY(float angle);
	void rotateX(float angle);

	void updateViewMat();

	XMMATRIX getViewMat();
	XMMATRIX getProjMat();

private:
	float FOVY;
	float aspect;
	float nearPlane;
	float farPlane;

	XMFLOAT3 position{ 0, 0, 0 };
	XMFLOAT3 up{ 0, 1, 0 };
	XMFLOAT3 forward{ 0, 0, 1 };
	XMFLOAT3 right{ 1, 0, 0 };

	XMFLOAT4X4 viewMat;
	XMFLOAT4X4 projMat;
};