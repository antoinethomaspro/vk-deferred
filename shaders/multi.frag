#version 450

layout (location = 0) in vec2 outUV;
layout (location = 0) out vec4 FragColor;

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverseView;
} ubo1;

layout (set=1, binding = 0) uniform sampler2D gPosition;
layout (set=1, binding = 1) uniform sampler2D gNormal;
layout (set=1, binding = 2) uniform sampler2D gAlbedoSpec;

vec3 fragPosWorld = texture(gPosition, outUV).xyz;
vec3 norm = texture(gNormal, outUV).xyz; 
vec3 vertColor = texture(gAlbedoSpec, outUV).xyz;

const vec3 DIRECTION_TO_LIGHT = normalize(vec3(0.0, 0.0, -10.0));
const float ambientStrength  = 0.01;
const vec3 lightColor = vec3(1.0);
float specularStrength = 0.5;

void main()
{
    vec3 ambient = ambientStrength * vertColor; 
    FragColor = vec4(ambient, 1.0);
}
