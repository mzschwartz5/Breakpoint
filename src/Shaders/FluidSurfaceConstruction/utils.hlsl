uint to1D(uint3 index, uint3 dimensions) {
    return index.x + (index.y * dimensions.x) + (index.z * dimensions.x * dimensions.y);
}

// Use sparingly, prefer having things in 1D if possible. (Modulus and division are expensive)
uint3 to3D(uint index, uint3 dimensions) {
    uint x = index % dimensions.x;
    uint y = (index / dimensions.x) % dimensions.y;
    uint z = index / (dimensions.x * dimensions.y);
    return uint3(x, y, z);
}

float cubic(float x) {
    return x * x * x;
}