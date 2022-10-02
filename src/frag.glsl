
#version 150

in vec3 v_normal;
in vec3 v_position;
in float v_tex_coord;

out vec4 color;

uniform vec3 u_light;
uniform sampler1D tex;

// Some of these parameters, like specular color or shininess, could be
// moved into uniforms, or they're probably fine as defaults

const vec4 specular_color = vec4(0.1, 0.1, 0.1, 1.0);

vec4 diffuse_color = texture(tex, v_tex_coord);
vec4 ambient_color = diffuse_color * 0.1;

void main()
{
	float diffuse =
			max(dot(normalize(v_normal), normalize(u_light)), 0.0);

	vec3 camera_dir = normalize(-v_position);
	vec3 half_dir = normalize(normalize(u_light) + camera_dir);
	float specular =
			pow(max(dot(half_dir, normalize(v_normal)), 0.0), 40.0);

	color = ambient_color +  diffuse *  diffuse_color
	                      + specular * specular_color;
}

