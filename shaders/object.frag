#version 450

layout(location = 0) in vec3 FragPos;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec3 lightPos;

layout(location = 0) out vec4 outColor;

const vec3 objectColor = vec3(1.0, 0.0, 0.0);

void main() {
    // ambient
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float ambientStrength = 0.02;
    vec3 ambient = ambientStrength * lightColor;

    //diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);  
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 result = (ambient + diffuse) * objectColor;
    outColor = vec4(result, 1.0);
}
