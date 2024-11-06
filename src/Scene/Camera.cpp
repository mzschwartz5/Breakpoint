#include "Camera.h"

Camera::Camera() {
	setFOV(0.25f * XM_PI, float(SCREEN_WIDTH)/SCREEN_HEIGHT, 0.1f, 1000.0f);
	XMMATRIX P = XMMatrixPerspectiveFovLH(FOVY, aspect, nearPlane, farPlane);
	XMStoreFloat4x4(&projMat, P);
}

void Camera::setFOV(float p_FOVY, float p_aspect, float p_nearPlane, float p_farPlane) {
	FOVY = p_FOVY;
	aspect = p_aspect;
	nearPlane = p_nearPlane;
	farPlane = p_farPlane;
}

void Camera::rotateY(float angle) {
	XMMATRIX R = XMMatrixRotationY(angle);

	XMStoreFloat3(&right, XMVector3TransformNormal(XMLoadFloat3(&right), R));
	XMStoreFloat3(&up, XMVector3TransformNormal(XMLoadFloat3(&up), R));
	XMStoreFloat3(&forward, XMVector3TransformNormal(XMLoadFloat3(&forward), R));
}

void Camera::rotateX(float angle) {
	XMMATRIX R = XMMatrixRotationAxis(XMLoadFloat3(&right), angle);

	XMStoreFloat3(&up, XMVector3TransformNormal(XMLoadFloat3(&up), R));
	XMStoreFloat3(&forward, XMVector3TransformNormal(XMLoadFloat3(&forward), R));
}

void Camera::updateViewMat() {
	XMVECTOR R = XMLoadFloat3(&right);
	XMVECTOR U = XMLoadFloat3(&up);
	XMVECTOR F = XMLoadFloat3(&forward);
	XMVECTOR P = XMLoadFloat3(&position);

	F = XMVector3Normalize(F);
	U = XMVector3Normalize(XMVector3Cross(F, R));
	R = XMVector3Cross(U, F);

	float x = -XMVectorGetX(XMVector3Dot(P, R));
	float y = -XMVectorGetX(XMVector3Dot(P, U));
	float z = -XMVectorGetX(XMVector3Dot(P, F));

	XMStoreFloat3(&right, R);
	XMStoreFloat3(&up, U);
	XMStoreFloat3(&forward, F);

	viewMat(0, 0) = right.x;
	viewMat(1, 0) = right.y;
	viewMat(2, 0) = right.z;
	viewMat(3, 0) = -position.x;

	viewMat(0, 1) = up.x;
	viewMat(1, 1) = up.y;
	viewMat(2, 1) = up.z;
	viewMat(3, 1) = -position.y;

	viewMat(0, 2) = forward.x;
	viewMat(1, 2) = forward.y;
	viewMat(2, 2) = forward.z;
	viewMat(3, 2) = -position.z;

	viewMat(0, 3) = 0.0f;
	viewMat(1, 3) = 0.0f;
	viewMat(2, 3) = 0.0f;
	viewMat(3, 3) = 1.0f;
}

XMMATRIX Camera::getViewMat() {
	return XMLoadFloat4x4(&viewMat);
}

XMMATRIX Camera::getProjMat() {
	return XMLoadFloat4x4(&projMat);
}
