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

const vec3 DIRECTION_TO_LIGHT = normalize(vec3(0.0, 0.0, -10.0));
const float ambientStrength  = 0.02;
const vec3 lightColor = vec3(1.0);
float specularStrength = 0.5;

void main()
{
    // Visualize the gPosition buffer
    vec3 FragPos = texture(gPosition, outUV).rgb;
    vec3 Normal = texture(gNormal, outUV).rgb;
    vec3 Albedo = texture(gAlbedoSpec, outUV).rgb;

    //----------computing light------------//
    vec3 ambient = ambientStrength * lightColor; 
    vec3 lightPos = ubo1.model[3].xyz;
    vec3 norm = Normal;
    vec3 lightDir = normalize(lightPos - FragPos);   

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    //specular
    vec3 cameraPosWorld = ubo1.inverseView[3].xyz;
    vec3 viewDir = normalize(cameraPosWorld - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;  

    vec3 result1 = (ambient + diffuse + specular) * Albedo;
    FragColor = vec4(result1, 1.0);
    
}
