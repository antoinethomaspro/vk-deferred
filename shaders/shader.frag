#version 450

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverseView;
} ubo1;

layout(location = 0) in vec3 vertColor;
layout(location = 1) in vec3 fragPosWorld;
layout(location = 2) in vec3 fragNormalWorld;

layout (location = 0) out vec3 gPosition; //white
layout (location = 1) out vec3 gNormal; //red
layout (location = 2) out vec4 gAlbedoSpec; //blue

void main() { 
    gPosition = fragPosWorld;
    gNormal = normalize(fragNormalWorld);
    gAlbedoSpec.rgb = vertColor; 
    gAlbedoSpec.a = 0.5;
}
