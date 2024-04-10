#version 450

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverseView;
} ubo;

layout(push_constant) uniform Push{
    vec3 offset;
    vec3 color;
} push;


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;


mat4 translationMatrix(vec3 translation) {
    return mat4(
        vec4(7.0, 0.0, 0.0, 0.0),
        vec4(0.0, 7.0, 0.0, 0.0),
        vec4(0.0, 0.0, 7.0, 0.0),
        vec4(translation, 1.0)
    );
}

mat4 matrix = translationMatrix(push.offset);

void main() {
    gl_Position = ubo.proj * ubo.view * matrix * vec4(inPosition, 1.0);
}
