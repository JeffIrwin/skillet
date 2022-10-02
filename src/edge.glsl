
#version 150
out vec4 color;

// This could be a uniform
const vec4 edge_color = vec4(0.0, 0.0, 0.0, 1.0);

void main()
{
	color = edge_color;
}

