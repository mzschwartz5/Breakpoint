#pragma once
#include <Alembic/AbcCoreFactory/All.h>
#include <Alembic/AbcGeom/All.h>
#include "Support/WinInclude.h"

using namespace Alembic::Abc;
using namespace Alembic::AbcGeom;
using namespace DirectX;

class AlembicLoader {

public:
    AlembicLoader(const AlembicLoader&) = delete;
    const XMFLOAT3* getParticlesForNextFrame() ;
    uint32_t getMaxParticleCount() const;
    uint32_t getParticleCountForCurrentFrame() const;

    static AlembicLoader& getInstance() {
        static AlembicLoader instance;
        return instance;
    }

private:
    AlembicLoader();
    ~AlembicLoader() = default;

    void visitObject(const IObject& obj);
    XMFLOAT4X4 getTransform(const IObject& obj) const;

    int frame = 0;
    int frameCount = 0;
    uint32_t maxParticleCount = 0;
    std::vector<uint32_t> particleCounts;
    std::vector<uint32_t> particleOffsets;
    std::vector<XMFLOAT3> particles;

};