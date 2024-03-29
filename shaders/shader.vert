#version 450

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverseView;
} ubo1;

layout (set=1, binding=0) uniform UniformBufferObject2 {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverseView;
} ubo2;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;

layout(location = 0) out vec3 vertColor;
layout(location = 1) out vec3 fragPosWorld; //needed to calculate the direction of the light for each fragment (we get the normal position)
layout(location = 2) out vec3 fragNormalWorld; //again, we need to interpolate the normal so each fragment has its own normal



void main() {
    gl_Position = ubo1.proj * ubo1.view * ubo2.model * vec4(inPosition, 1.0);
    
    // temporary: only correct in certain situations!
    fragNormalWorld = normalize(mat3(ubo2.model) * normal);

    fragPosWorld = (ubo2.model * vec4(inPosition, 1.0)).xyz;
    vertColor = inColor;
}
