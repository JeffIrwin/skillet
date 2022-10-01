
//==============================================================================

//use crate::colormaps::*;

pub struct Background
{
	pub colormap: glium::texture::SrgbTexture1d,
}

fn get_bg_colormap(display: &glium::Display) -> glium::texture::SrgbTexture1d
{
	// Define the colormap for the gradient background.  It has to be saved as
	// a texture to be able to use gamma correction (sRGB).  Colors are based on
	// jekyll cayman theme.
	let cmap = vec!
		[
			 21u8,  87u8, 154u8, 255u8,
			 21u8, 154u8,  87u8, 255u8,
		];

	let image = glium::texture::RawImage1d::from_raw_rgba(cmap);
	glium::texture::SrgbTexture1d::new(display, image).unwrap()
}

//==============================================================================

impl Background
{
	pub fn new(display: &glium::Display) -> Background
	{
		let colormap = get_bg_colormap(&display);
		Background
		{
			colormap: colormap,
		}
	}
}

//==============================================================================

