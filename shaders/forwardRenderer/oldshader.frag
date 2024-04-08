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

layout (location = 0) out vec4 fragColor; 

struct Light {
    vec3 position;
    vec3 color;
};

const int NUM_LIGHTS = 2; // Define the number of lights

const float ambientStrength  = 0.01;
const vec3 ambientColor = vec3(1.0);
float specularStrength = 0.5;

//variables for all lights: 
vec3 norm = normalize(fragNormalWorld);
vec3 cameraPosWorld = ubo1.inverseView[3].xyz;
vec3 viewDir = normalize(cameraPosWorld - fragPosWorld);


vec3 computePointLight(vec3 lightPos, vec3 lightColor)
{
    //attenuation factor
    float distance = length(lightPos - fragPosWorld);
    float attenuation = 1.0 / (1.0+ 0.07 * distance + 1.8 * (distance * distance));  

    vec3 lightDir = normalize(lightPos - fragPosWorld);

    //diff
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    //specular
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor; 

    diffuse  *= attenuation;
    specular *= attenuation; 

    return (diffuse + specular) * vertColor; 
}

void main() { 
    vec3 finalColor = vec3(0.0); // Initialize final color

    // Define an array of lights and initialize it with one hardcoded light
    Light lights[NUM_LIGHTS];
    lights[0].position = vec3(0.0, 0.0, -2.0); // Hardcoded light position
    lights[0].color = vec3(1.0, 0.0, 0.0); // Red color
    lights[1].position = vec3(0.0, 0.0, -1.0); // Hardcoded light position
    lights[1].color = vec3(0.0, 1.0, 0.0); // Red color

    for (int i = 0; i < NUM_LIGHTS; ++i) {
        vec3 lightPos = lights[i].position;
        vec3 lightColor = lights[i].color;

        // Compute light contribution for this point light
        vec3 lightContribution = computePointLight(lightPos, lightColor);

        // Accumulate the light contribution
        finalColor += lightContribution;
    }

    vec3 ambient = ambientStrength * ambientColor; 
    // Add ambient light to the final color
    fragColor = vec4(ambient + finalColor, 1.0);
    
}
