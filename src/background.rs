
//==============================================================================

use crate::consts::*;

#[derive(Copy, Clone, Debug)]
pub struct Vert2
{
	// 2D node for background
	position2: [f32; N2],
	tex_coord: f32,
}
implement_vertex!(Vert2, position2, tex_coord);

pub struct Background
{
	pub colormap: glium::texture::SrgbTexture1d,
	pub program : glium::Program,
	pub vertices: glium::VertexBuffer<Vert2>,
	pub indices : glium::IndexBuffer<u32>,
}

//==============================================================================

fn get_colormap(facade: &dyn glium::backend::Facade) -> glium::texture::SrgbTexture1d
{
	// Define the colormap for the gradient background.  It has to be saved as
	// a texture to be able to use gamma correction (sRGB).  Colors are based on
	// jekyll cayman theme.
	//
	// c.f. colormaps.rs
	//
	let cmap = vec!
		[
			 21u8,  87u8, 154u8, 255u8,
			 21u8, 154u8,  87u8, 255u8,
		];

	let image = glium::texture::RawImage1d::from_raw_rgba(cmap);
	glium::texture::SrgbTexture1d::new(facade, image).unwrap()
}

//==============================================================================

impl Background
{
	pub fn new(facade: &dyn glium::backend::Facade) -> Background
	{
		let colormap = get_colormap(facade);

		let vertex_shader_src = r#"
			#version 150

			in vec2 position2;
			in float tex_coord;
			out float v_tex_coord;

			void main() {
				v_tex_coord = tex_coord;
				gl_Position = vec4(position2, 0, 1.0);
			}
		"#;

		let fragment_shader_src = r#"
			#version 150

			in float v_tex_coord;
			out vec4 color;

			uniform sampler1D bg_tex;

			void main() {
				color = texture(bg_tex, v_tex_coord);
			}
		"#;

		// c.f. shaders.rs
		let program = glium::Program::from_source(facade, vertex_shader_src,
			fragment_shader_src, None).unwrap();

		// background vertices
		let verts = vec!
			[
				Vert2 { position2: [-1.0, -1.0], tex_coord: 0.5, },
				Vert2 { position2: [ 1.0, -1.0], tex_coord: 1.0, },
				Vert2 { position2: [ 1.0,  1.0], tex_coord: 0.5, },
				Vert2 { position2: [-1.0,  1.0], tex_coord: 0.0, },
			];

		Background
		{
			colormap: colormap,
			program : program,
			vertices: glium::VertexBuffer::new(facade, &verts).unwrap(),
			indices : glium::IndexBuffer::new(facade,
				glium::index::PrimitiveType::TrianglesList,
				&[
					0, 1, 2,
					2, 3, 0 as u32
				]).unwrap(),
		}
	}
}

//==============================================================================

