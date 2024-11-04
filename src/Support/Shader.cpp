#include "Shader.h"

Shader::Shader(std::string_view name) {
    static std::filesystem::path shaderDir;

    if (shaderDir.empty()) {
        wchar_t moduleFileName[MAX_PATH];
        GetModuleFileNameW(nullptr, moduleFileName, MAX_PATH);
        shaderDir = moduleFileName;
        shaderDir.remove_filename();
    }

    std::ifstream shaderIn(shaderDir / name, std::ios::binary);

    if (shaderIn.is_open()) {
        shaderIn.seekg(0, std::ios::end);
        size = shaderIn.tellg();
        shaderIn.seekg(0, std::ios::beg);
        data = malloc(size);
        if (data) {
            shaderIn.read((char*)data, size);
        }
    }
}

Shader::~Shader() {
    if (data) {
        free(data);
    }
}