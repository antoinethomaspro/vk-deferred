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

mat4 translateMatrix(vec3 translation) {
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(translation, 1.0)
    );
}

void main() {

     vec3 translation1 = vec3(0.0, 0.0, 0.0); // Example translation vector
    mat4 translationMatrix1 = translateMatrix(translation1);

    vec3 translation2 = vec3(0.0, 1.0, -2.0); // Example translation vector
    mat4 translationMatrix2 = translateMatrix(translation2);

     vec3 translation3 = vec3(0.0, 0.0, -4.0); // Example translation vector
    mat4 translationMatrix3 = translateMatrix(translation3);

     // Hardcoded transformation matrices for two instances
    mat4 mvp_matrix[3];
    mvp_matrix[0] = translationMatrix1; // Identity matrix for instance 0
    mvp_matrix[1] = translationMatrix2; // Identity matrix for instance 1
    mvp_matrix[2] = translationMatrix3; // Identity matrix for instance 2

    mat4 transformationMatrix = mvp_matrix[gl_InstanceIndex];

    gl_Position = ubo1.proj * ubo1.view * transformationMatrix * vec4(inPosition, 1.0);
    
    // temporary: only correct in certain situations!
    fragNormalWorld = normalize(mat3(transformationMatrix) * normal);

    fragPosWorld = (transformationMatrix * vec4(inPosition, 1.0)).xyz;
    vertColor = inColor;
}
