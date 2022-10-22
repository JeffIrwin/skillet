
//==============================================================================

// Standard
use std::env;
use std::path::PathBuf;
use std::rc::Rc;

//****************

// This crate, included in lib
use skillet::*;
use crate::consts::*;
use crate::model;
use crate::model::RenderModel;

// Not included in lib
pub mod app;
pub mod background;

use app::State;

//****************
// 3P(s)
//****************

#[macro_use]
extern crate glium;

use glium::glutin::event_loop::EventLoop;

//==============================================================================

fn main()
{

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
	let model = Box::new(model::import(file_path));

	let event_loop = EventLoop::new();

	// Ownership of the Display is shared between the RenderModel and the State,
	// so it's in an Rc.  It can't be singly-owned as far as I can tell, because
	// RenderModel uses it cast as a facade, while State uses it directly as
	// Display :(
	let display = Rc::new(app::display(&event_loop));

	let render_model = Box::new(RenderModel::new(model, display.clone()));

	let mut state = State::new(render_model, display.clone());

	////let system = glium_text::TextSystem::new(&*display);
	////let system = glium_text::TextSystem::new::<dyn glium::backend::Facade>(&*display.borrow());
	//let system = glium_text::TextSystem::new::<glium::Display>(&*display.borrow());

	println!("{}:  Starting main loop", ME);
	println!();

	event_loop.run(move |event, _, control_flow|
	{
		app::main_loop(&event, control_flow, &mut state);
	});
}

//==============================================================================

