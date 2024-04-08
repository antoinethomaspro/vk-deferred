#version 450

layout(location = 0) out vec4 FragColor;

layout (set=0, binding=0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverseView;
} ubo1;

struct Light {
    vec3 position;
    vec3 color;
};

layout (set=1, binding = 0) uniform sampler2D gPosition;
layout (set=1, binding = 1) uniform sampler2D gNormal;
layout (set=1, binding = 2) uniform sampler2D gAlbedoSpec;

vec2 gScreenSize = vec2(800.0, 600.0);

vec2 CalcTexCoord()
{
   return gl_FragCoord.xy / gScreenSize;
} 

//sample the gbuffer textures
vec2 TexCoord = CalcTexCoord();
vec3 fragPosWorld = texture(gPosition, TexCoord).xyz;
vec3 norm = texture(gNormal, TexCoord).xyz; // !!! The normals are already normalized
vec3 vertColor = texture(gAlbedoSpec, TexCoord).xyz;

//other global operations
vec3 cameraPosWorld = ubo1.inverseView[3].xyz;
vec3 viewDir = normalize(cameraPosWorld - fragPosWorld);

float specularStrength = 0.5;

void main()
{
    //define the point light processed
    Light pLight;
    pLight.position = vec3(0.0, 0.0, -1.0); //same as our light object
    pLight.color = vec3(0.0, 1.0, 0.0); // red

    //attenuation factor
    float distance = length(pLight.position - fragPosWorld);
    float attenuation = 1.0 / (1.0+ 0.07 * distance + 1.8 * (distance * distance));  

    vec3 lightDir = normalize(pLight.position - fragPosWorld);

    //diff
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * pLight.color;
    
    //specular
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * pLight.color; 

    diffuse  *= attenuation;
    specular *= attenuation; 

    vec3 finalColor =  vec3(diffuse + specular) * vertColor; 
    vec3 red = vec3(1.0);
    FragColor = vec4(finalColor, 1.0);
} 