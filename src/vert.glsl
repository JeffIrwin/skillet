
#version 150

in vec3 position;
in vec3 normal;
in float tex_coord;

out vec3 v_normal;
out vec3 v_position;
out float v_tex_coord;

uniform mat4 perspective;
uniform mat4 view;
uniform mat4 model_mat;
uniform mat4 world;

void main()
{
	v_tex_coord = tex_coord;
	mat4 modelview = view * world * model_mat;
	v_normal = transpose(inverse(mat3(modelview))) * normal;
	gl_Position = perspective * modelview * vec4(position, 1.0);
	v_position = gl_Position.xyz / gl_Position.w;
}

