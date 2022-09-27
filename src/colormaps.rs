
//==============================================================================

use serde_json;
const CMAPS: &str = include_str!("../res/colormaps.json");

pub fn get_colormap(index: &mut usize, display: &glium::Display) -> glium::texture::SrgbTexture1d
{
	// Define the colormap by loading an indexed map from a json array
	//
	// TODO: currate resource file.  Some maps need extra interp points to look
	// good, some are in color spaces that I can't handle properly, and some
	// just don't look that interesting

	//println!("CMAPS = {}", CMAPS);

	let cmaps: serde_json::Value = serde_json::from_str(CMAPS).unwrap();

	let cmaps_len = cmaps.as_array().unwrap().len();
	if *index >= cmaps_len
	{
		*index = 0;
	}

	//println!("CMAPS[{}] = {}", *index, cmaps[*index]);

	let xrgb = &cmaps[*index]["RGBPoints"];
	let xrgb_len = xrgb.as_array().unwrap().len();

	//println!("xrgb = {}", xrgb);
	//println!("xrgb.len() = {}", xrgb_len);

	let mut cmap = Vec::<u8>::with_capacity(xrgb_len);

	// Format is slightly different.  Ignore x value from xrgb, add alpha value,
	// and scale everything from [0, 1] to [0, 255]

	let cmax = 255.0;
	for i in (0 .. xrgb_len).step_by(4)
	{
		cmap.push((xrgb[i+1].as_f64().unwrap() * cmax) as u8); // R
		cmap.push((xrgb[i+2].as_f64().unwrap() * cmax) as u8); // G
		cmap.push((xrgb[i+3].as_f64().unwrap() * cmax) as u8); // B

		// Alpha
		cmap.push(cmax as u8);
	}

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

