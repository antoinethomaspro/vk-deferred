#version 450

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout (set=1, binding=0) uniform UniformBufferObject2 {
    mat4 model;
    mat4 view;
    mat4 proj;
} lol;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;

layout(location = 0) out vec3 fragColor;

const vec3 DIRECTION_TO_LIGHT = normalize(vec3(1.0, -3.0, -1.0));
const float AMBIENT = 0.02;

float lightIntensity = AMBIENT + max(dot(normal, DIRECTION_TO_LIGHT), 0);


void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
  //  fragColor = lol.model[0].xyz;
    fragColor =  inColor;
}
