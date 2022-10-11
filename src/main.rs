
//==============================================================================

// Standard
use std::env;
use std::io::Cursor;

//****************

// This crate, included in lib
use skillet::*;
use crate::consts::*;
use crate::math::*;
use crate::model::*;
use crate::utils::*;

// Not included in lib
pub mod app;
pub mod background;

//****************
// 3P(s)
//****************

#[macro_use]
extern crate glium;

use glium::{glutin, glutin::event_loop::EventLoop};

//==============================================================================

fn main()
{
	use std::path::PathBuf;

	println!();
	println!("{}:  Starting main()", ME);
	println!("{}:  {}", ME, mev!());
	println!();

	let exe = utils::get_exe_base("skillet.exe");

	// Switch to clap if args become more complicated than a single filename
	let args: Vec<String> = env::args().collect();
	if args.len() < 2
	{
		println!("Error: bad command-line arguments");
		println!("Usage:");
		println!();
		println!("\t{} FILE.VTK", exe);
		println!();
		return;
	}

	let file_path = PathBuf::from(args[1].clone());

	let model = Box::new(import(file_path));

	// TODO: refactor to window init fn

	let event_loop = EventLoop::new();

	// TODO: cleanup some of these use paths

	// include_bytes!() statically includes the file relative to this source
	// path at compile time
	let icon = image::load(Cursor::new(&include_bytes!("../res/icon.png")),
			image::ImageFormat::Png).unwrap().to_rgba8();
	let winicon = Some(glutin::window::Icon::from_rgba(icon.to_vec(),
			icon.dimensions().0, icon.dimensions().1).unwrap());

	let wb = glutin::window::WindowBuilder::new()
		.with_title(mev!())
		.with_window_icon(winicon)
		.with_maximized(true);
		//.with_inner_size(glutin::dpi::LogicalSize::new(1960.0, 1390.0))
		//.with_position(glutin::dpi::LogicalPosition::new(0, 0));
		//// ^ this leaves room for an 80 char terminal on my main monitor

	let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
	let display = glium::Display::new(wb, cb, &event_loop).unwrap();

	let mut s = app::State::new(&display);

	//****************

	// Get point xyz bounds.  TODO: this could be moved into State construction

	let (xmin, xmax) = get_bounds(&(model.points.iter().skip(0)
			.step_by(ND).copied().collect::<Vec<f32>>()));

	let (ymin, ymax) = get_bounds(&(model.points.iter().skip(1)
			.step_by(ND).copied().collect::<Vec<f32>>()));

	let (zmin, zmax) = get_bounds(&(model.points.iter().skip(2)
			.step_by(ND).copied().collect::<Vec<f32>>()));

	let xc = 0.5 * (xmin + xmax);
	let yc = 0.5 * (ymin + ymax);
	let zc = 0.5 * (zmin + zmax);

	s.cen = [xc, yc, zc];

	s.diam = norm(&sub(&[xmax, ymax, zmax], &[xmin, ymin, zmin]));

	println!("x in [{}, {}]", ff32(xmin), ff32(xmax));
	println!("y in [{}, {}]", ff32(ymin), ff32(ymax));
	println!("z in [{}, {}]", ff32(zmin), ff32(zmax));
	println!();

	let mut render_model = RenderModel::new(model, &display);

	// View must be initialized like this, because subsequent rotations are
	// performed about its fixed coordinate system.  Set eye from model bounds.
	// You could do some trig here on fov to guarantee whole model is in view,
	// but it's pretty close as is except for possible extreme cases

	s.eye = [0.0, 0.0, zmax + s.diam];
	s.view = view_matrix(&s.eye, &app::DIR, &app::UP);

	// Initial pan to center
	s.world = translate_matrix(&s.world, &neg(&s.cen));
	s.cen = [0.0; ND];

	println!("{}:  Starting main loop", ME);
	println!();

	event_loop.run(move |event, _, control_flow|
	{
		app::main_loop(&event, control_flow, &mut s, &mut render_model, &display);
	});
}

//==============================================================================

