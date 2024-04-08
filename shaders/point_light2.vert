#version 450

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverseView;
} ubo;


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;

mat4 translateMatrix(vec3 translation) {
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(translation, 1.0)
    );
}

void main() {

  //  vec3 translation1 = vec3(0.0, 0.0, 0.0); // Example translation vector
  //  mat4 translationMatrix1 = translateMatrix(translation1);


    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
}
