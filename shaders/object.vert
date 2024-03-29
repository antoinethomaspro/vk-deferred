#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec2 uv;

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


layout(location = 0) out vec3 FragPos;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec3 lightPos;


//float lightIntensity = AMBIENT + max(dot(normal, DIRECTION_TO_LIGHT), 0);

void main() {

    gl_Position = ubo.proj * ubo.view  * lol.model *  vec4(inPosition, 1.0);
    FragPos = vec3(lol.model * vec4(inPosition, 1.0));
    Normal = lol.model * aNormal;

    lightPos = ubo.model[3].xyz;
   
}
