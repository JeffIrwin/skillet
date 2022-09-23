
//==============================================================================

pub fn get_colormap(display: &glium::Display) -> glium::texture::SrgbTexture1d
{
	// Define the colormap.  Hard-code for now
	//
	// TODO: load from res file w/ include_bytes like icon

	// Red to Blue Rainbow (RGBA)
	let cmap = vec!
		[
			  0u8,   0u8,   255u8, 255u8,
			  0u8,  64u8,   255u8, 255u8,
			  0u8, 128u8,   255u8, 255u8,
			  0u8, 192u8,   255u8, 255u8,
			  0u8, 255u8,   255u8, 255u8,
			  0u8, 255u8,   192u8, 255u8,
			  0u8, 255u8,   128u8, 255u8,
			  0u8, 255u8,    64u8, 255u8,
			  0u8, 255u8,     0u8, 255u8,
			 64u8, 255u8,     0u8, 255u8,
			128u8, 255u8,     0u8, 255u8,
			192u8, 255u8,     0u8, 255u8,
			255u8, 255u8,     0u8, 255u8,
			255u8, 192u8,     0u8, 255u8,
			255u8, 128u8,     0u8, 255u8,
			255u8,  64u8,     0u8, 255u8,
			255u8,   0u8,     0u8, 255u8,
		];

	//// Black-Body Radiation.  TODO: this probably needs to be interpolated and
	//// expanded
	//let cmap = vec!
	//	[
	//		  0u8,   0u8,   0u8, 255u8,
	//		230u8,   0u8,   0u8, 255u8,
	//		230u8, 230u8,   0u8, 255u8,
	//		255u8, 255u8, 255u8, 255u8,
	//	];

	let image = glium::texture::RawImage1d::from_raw_rgba(cmap);
	glium::texture::SrgbTexture1d::new(display, image).unwrap()
}

//==============================================================================

pub fn get_bg_colormap(display: &glium::Display) -> glium::texture::SrgbTexture1d
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

