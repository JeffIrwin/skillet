
//==============================================================================

// Standard
use std::env;
use std::io::Cursor;
use std::rc::Rc;

//****************

// This crate, included in lib
use skillet::*;
use crate::consts::*;
use crate::model::*;

// Not included in lib
pub mod app;
pub mod background;

use app::State;

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

	// Ownership of the Model is transferred to the RenderModel later, so it's
	// in a Box
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

	// Ownership of the Display is shared between the RenderModel and the State,
	// so it's in an Rc.  It can't be singly-owned as far as I can tell, because
	// RenderModel uses it cast as a facade, while State uses it directly as
	// Display :(
	let display = Rc::new(glium::Display::new(wb, cb, &event_loop).unwrap());

	let render_model = Box::new(RenderModel::new(model, display.clone()));

	let mut s = State::new(render_model, display);

	println!("{}:  Starting main loop", ME);
	println!();

	event_loop.run(move |event, _, control_flow|
	{
		app::main_loop(&event, control_flow, &mut s);
	});
}

//==============================================================================

