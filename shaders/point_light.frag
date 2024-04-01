#version 450

layout(location = 0) out vec4 FragColor;

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverseView;
} ubo1;

layout (set=1, binding = 0) uniform sampler2D gPosition;
layout (set=1, binding = 1) uniform sampler2D gNormal;
layout (set=1, binding = 2) uniform sampler2D gAlbedoSpec;

vec2 gScreenSize = vec2(800.0, 600.0);

vec2 CalcTexCoord()
{
   return gl_FragCoord.xy / gScreenSize;
} 

void main()
{
    vec2 TexCoord = CalcTexCoord();
    vec3 WorldPos = texture(gPosition, TexCoord).xyz;
    vec3 Normal = texture(gNormal, TexCoord).xyz;
    vec3 Color = texture(gAlbedoSpec, TexCoord).xyz;

    FragColor = vec4(Color, 1.0);
} 