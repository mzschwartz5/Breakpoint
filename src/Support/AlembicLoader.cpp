#include "AlembicLoader.h"
#include <filesystem>

AlembicLoader::AlembicLoader() {
    Alembic::AbcCoreFactory::IFactory factory;
    Alembic::AbcCoreFactory::IFactory::CoreType coreType;

    IArchive archive = factory.getArchive((std::filesystem::current_path() / "FluidBeach.abc").string(), coreType);
    visitObject(archive.getTop());
}

void AlembicLoader::visitObject(const IObject& obj) {
    // Print name and type of object
    std::cout << "Name: " << obj.getFullName() << std::endl;

    if (IPoints::matches(obj.getMetaData())) {
        // Get IPointsSchema
        IPoints points = IPoints(obj);
        IPointsSchema schema = points.getSchema();

        // Get number of samples
        frameCount = static_cast<int>(schema.getNumSamples());
        if (frameCount == 0) {
            return;
        }

        particleCounts.resize(frameCount);
        particleOffsets.resize(frameCount);

        XMFLOAT4X4 transform = getTransform(points.getParent());

        // Process each samples
        for (int i = 0; i < frameCount; ++i) {
            IPointsSchema::Sample sample;
            schema.get(sample, ISampleSelector((index_t)i));

            // Get particle positions
            P3fArraySamplePtr positions = sample.getPositions();
            particleCounts[i] = static_cast<uint32_t>(positions->size());
            particleOffsets[i] = i == 0 ? 0 : particleOffsets[i - 1] + particleCounts[i - 1];

            if (positions) {
                for (size_t j = 0; j < particleCounts[i]; ++j) {
                    const Imath::V3f& pos = (*positions)[j];
                    XMFLOAT4 position(pos.x, pos.y, pos.z, 0.0f);
                    XMVECTOR positionVector = XMLoadFloat4(&position);
                    XMMATRIX transformMatrix = XMLoadFloat4x4(&transform);
                    XMVECTOR transformedPosition = XMVector4Transform(positionVector, transformMatrix);
                    XMFLOAT4 transformedPositionFloat4;
                    XMStoreFloat4(&transformedPositionFloat4, transformedPosition);
                    particles.push_back(XMFLOAT3(transformedPositionFloat4.x, transformedPositionFloat4.y, transformedPositionFloat4.z));                }
            }
        }
        maxParticleCount = std::ranges::max(particleCounts);
    }

    // Process child objects
    for (size_t i = 0; i < obj.getNumChildren(); ++i) {
        visitObject(obj.getChild(i));
    }
}

XMFLOAT4X4 AlembicLoader::getTransform(const IObject& obj) const
{
    if (IXform::matches(obj.getMetaData())) {
        IXform xform(obj, kWrapExisting);
        XformSample sample;
        xform.getSchema().get(sample);

        const auto& alembicMatrix = sample.getMatrix();
        XMFLOAT4X4 transform;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                transform.m[i][j] = static_cast<float>(alembicMatrix[i][j]);
            }
        }

        return transform;
    }
    return XMFLOAT4X4{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

const XMFLOAT3* AlembicLoader::getParticlesForNextFrame()
{
    frame = (frame + 1) % frameCount;
    return particles.data() + particleOffsets[frame];
}

uint32_t AlembicLoader::getParticleCountForCurrentFrame() const
{
    return particleCounts[frame];
}

uint32_t AlembicLoader::getMaxParticleCount() const
{
    return maxParticleCount;
}

