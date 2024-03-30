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

layout (location = 0) out vec4 fragColor0; //white
layout (location = 1) out vec4 fragColor1; //red
layout (location = 2) out vec4 fragColor2; //blue

const vec3 DIRECTION_TO_LIGHT = normalize(vec3(0.0, 0.0, -10.0));
const float ambientStrength  = 0.02;
const vec3 lightColor = vec3(1.0);
float specularStrength = 0.5;

void main() { 

    vec3 ambient = ambientStrength * lightColor; 

    vec3 lightPos = ubo1.model[3].xyz;
    vec3 norm = normalize(fragNormalWorld);
    vec3 lightDir = normalize(lightPos - fragPosWorld);   

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    //vec3 result = (ambient + diff) * vertColor;
    //fragColor = vec4(result, 1.0);

    //specular
    vec3 cameraPosWorld = ubo1.inverseView[3].xyz;
    vec3 viewDir = normalize(cameraPosWorld - fragPosWorld);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;  

    vec3 result1 = (ambient + diffuse + specular) * vertColor;
    vec3 result2 = (ambient + diffuse + specular) * vec3(1.0, 0.0, 0.0);
    vec3 result3 = (ambient + diffuse + specular) * vec3(0.0, 0.0, 1.0);
    fragColor0 = vec4(result1, 1.0);
    fragColor1 = vec4(result2, 1.0);
    fragColor2 = vec4(result3, 1.0);
    
}
