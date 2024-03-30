#version 450

// // Vertex positions for a full-screen quad
// vec2 positions[4] = vec2[](
//     vec2(-1.0, -1.0),
//     vec2( 1.0, -1.0),
//     vec2(-1.0,  1.0),
//     vec2( 1.0,  1.0)
// );

// // Corresponding texture coordinates
// vec2 texCoords[4] = vec2[](
//     vec2(0.0, 0.0),
//     vec2(1.0, 0.0),
//     vec2(0.0, 1.0),
//     vec2(1.0, 1.0)
// );

layout(location = 0) out vec2 outUV;

void main() {
     outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0f + -1.0f, 0.0f, 1.0f);
}