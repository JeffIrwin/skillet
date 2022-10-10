
//==============================================================================

// Standard
use std::env;
use std::io::Cursor;

//****************

// This crate, included in lib
use skillet::*;
use crate::colormaps::*;
use crate::consts::*;
use crate::math::*;
use crate::model::*;
use crate::utils::*;

// Not included in lib
pub mod background;
use crate::background::Background;

//****************
// 3P(s)
//****************

#[macro_use]
extern crate glium;

use glium::{glutin, glutin::event_loop::EventLoop, glutin::event, Surface};

//==============================================================================

struct State
{
	// This is the global app state

	// Modifier keys
	pub ctrl : bool,
	pub shift: bool,

	// Mouse button states
	pub lmb: bool,
	pub mmb: bool,
	pub rmb: bool,

	// This is where transformations happen
	pub world: [[f32; NM]; NM],

	pub view : [[f32; NM]; NM],

	pub cen: [f32; ND],
	pub eye: [f32; ND],

	// Mouse position from last frame
	pub x0: f32,
	pub y0: f32,

	// Scroll wheel zoom factor
	pub scale_cum: f32,

	// Colormap index in JSON res file
	pub map_index: usize,

	pub colormap: glium::texture::SrgbTexture1d,

	pub diam: f32,
	pub display_diam: f32,

	pub bg: Background,

	pub face_program: glium::Program,
	pub edge_program: glium::Program,
}

impl State
{
	pub fn new(display: &glium::Display) -> State
	{
		let mut cmi = 0;

		State
		{
			ctrl : false,
			shift: false,

			lmb: false,
			mmb: false,
			rmb: false,

			world: identity_matrix(),
			view : identity_matrix(),

			cen: [0.0; ND],
			eye: [0.0; ND],

			x0: 0.0,
			y0: 0.0,

			scale_cum: 1.0,

			colormap: get_colormap(&mut cmi, &display),
			//colormap: glium::texture::SrgbTexture1d::new(),
			map_index: cmi,

			// This initial value doesn't matter.  It will get set correctly
			// after the first frame
			display_diam: 1920.0,
			diam: 0.0,

			bg: Background::new(display),

			face_program: shaders::face(display),
			edge_program: shaders::edge(display),
		}
	}
}

// View constants
const DIR: [f32; ND] = [0.0, 0.0, -1.0];
const UP : [f32; ND] = [0.0, 1.0,  0.0];

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

	let mut s = State::new(&display);

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
	s.view = view_matrix(&s.eye, &DIR, &UP);

	// Initial pan to center
	s.world = translate_matrix(&s.world, &neg(&s.cen));
	s.cen = [0.0; ND];

	println!("{}:  Starting main loop", ME);
	println!();

	event_loop.run(move |event, _, control_flow|
	{
		main_loop(&event, control_flow, &mut s, &mut render_model, &display);
	});
}

//==============================================================================

fn main_loop<T>
	(
		event       : &glutin::event::Event<'_, T>,
		control_flow: &mut glutin::event_loop::ControlFlow,
		s           : &mut State,
		rm          : &mut RenderModel,
		display     : &glium::Display,
	)
{
	let next_frame_time = std::time::Instant::now() +
			std::time::Duration::from_nanos(16_666_667);
	*control_flow =
			glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

	const PRESSED: glutin::event::ElementState
	             = glutin::event::ElementState::Pressed;

	match event
	{
		glutin::event::Event::WindowEvent { event, ..} => match event
		{
			glutin::event::WindowEvent::CloseRequested =>
			{

				println!("{}:  Exiting main()", ME);
				println!();

				*control_flow = glutin::event_loop::ControlFlow::Exit;
				return;

			},
			event::WindowEvent::ModifiersChanged(modifiers_state) =>
			{
				//println!("modifiers_state = {:?}", modifiers_state);
				s.ctrl  = modifiers_state.ctrl ();
				s.shift = modifiers_state.shift();
			},
			glutin::event::WindowEvent::MouseInput  {state, button, ..} =>
			{
				//println!("state, button = {:?}, {:?}", state, button);

				match button
				{
					glutin::event::MouseButton::Left =>
					{
						s.lmb = *state == PRESSED;
					},
					glutin::event::MouseButton::Right =>
					{
						s.rmb = *state == PRESSED;
					},
					glutin::event::MouseButton::Middle =>
					{
						s.mmb = *state == PRESSED;
					},
					_ => ()
				}
			},
			glutin::event::WindowEvent::CursorMoved {position, ..} =>
			{
				//println!("position = {:?}", position);

				let x = position.x as f32;
				let y = position.y as f32;

				if s.lmb
				{
					// Rotate about axis within the xy screen plane
					//
					// TODO: handle shift-lmb as z rotation

					// Right-hand normal to drag direction
					let mut u = [-(y - s.y0), -(x - s.x0), 0.0];

					let norm = norm(&u);
					u[0] /= norm;
					u[1] /= norm;
					// z is zero, no need to normalize

					let sensitivity = 0.0035;
					let theta = sensitivity * norm;

					// Push translation to model center, apply rotation,
					// then pop trans
					s.world = translate_matrix(&s.world, &neg(&s.cen));
					s.world = rotate_matrix   (&s.world, &u, theta);
					s.world = translate_matrix(&s.world, &s.cen);

				}
				else if s.mmb
				{
					// xy pan

					//println!("mmb drag");

					let sensitivity = 1.5 * s.diam //* s.scale_cum
							/ s.display_diam;

					let dx =  sensitivity * (x - s.x0);// / display_h;
					let dy = -sensitivity * (y - s.y0);// / display_w;

					let tran = [dx, dy, 0.0];

					s.world = translate_matrix(&s.world, &tran);

					// Panning moves rotation center too.  add() returns
					// Vec, so we have to try_into() and unwrap() to array.
					s.cen = add(&s.cen, &tran).try_into().unwrap();
				}
				else if s.rmb
				{
					// z pan (eye motion zoom)
					//
					// This uses the opposite sign convention of ParaView,
					// but I think it feels more consistent with the scroll
					// wheel action:  scrolling up has a similar effect as
					// rmb dragging up

					let dz = y - s.y0;

					let sensitivity = 0.003;
					s.eye[2] += sensitivity * s.scale_cum * s.diam * dz;
					s.view = view_matrix(&s.eye, &DIR, &UP);
				}

				s.x0 = x;
				s.y0 = y;
			},
			glutin::event::WindowEvent::MouseWheel {delta, ..} =>
			{
				// Scroll scaling zoom
				//
				// ParaView actually has two ways to "zoom": (1) RMB-drag
				// moves the eye of the view, while (2) the scroll wheel
				// scales the world

				//println!("delta = {:?}", delta);

				let dz = match delta
				{
					glutin::event::MouseScrollDelta::LineDelta(_a, b) =>
					{
						//println!("a b = {} {}", a, b);
						*b
					},
					//glutin::event::MouseScrollDelta::PixelDelta(p) =>
					//{
					//	println!("p = {:?}", p);
					//	//unimplemented!()
					//},
					_ => (0.0)
				};

				// This sign convention matches ParaView, although the
				// opposite scroll/zoom convention does exist

				let sensitivity = 0.1;
				let scale = (sensitivity * dz).exp();
				s.scale_cum *= scale;

				//println!("scale = {}", scale);

				s.world = scale_matrix(&s.world, scale);
				s.cen   = scale_vec(&s.cen, scale).try_into().unwrap();
			},
			glutin::event::WindowEvent::KeyboardInput {input, ..} =>
			{
				//println!("input = {:?}", input);

				let warp_increment = 0.1;

				if s.ctrl && input.state == PRESSED
				{
					match input.virtual_keycode.unwrap()
					{
						event::VirtualKeyCode::W =>
						{
							//println!("Ctrl+W");
							rm.warp_factor -= warp_increment;
							rm.warp(display);
						}
						_ => {}
					}
				}
				else if s.shift && input.state == PRESSED
				{
					match input.virtual_keycode.unwrap()
					{
						event::VirtualKeyCode::W =>
						{
							//println!("Shift+W");
							rm.warp_factor += warp_increment;
							rm.warp(display);
						}
						_ => {}
					}
				}
				else if input.state == PRESSED
				{
					match input.virtual_keycode.unwrap()
					{
						// TODO: parameterize keycodes

						event::VirtualKeyCode::C =>
						{
							let name;
							if rm.dindex < rm.m.point_data.len()
							{
								rm.comp = (rm.comp + 1)
									% rm.m.point_data[rm.dindex].num_comp;
								rm.bind_point_data(display);
								name = &rm.m.point_data[rm.dindex].name;
							}
							else
							{
								let cindex = rm.dindex - rm.m.point_data.len();
								rm.comp = (rm.comp + 1)
									% rm.m.cell_data[cindex].num_comp;
								rm.bind_cell_data(display);
								name = &rm.m.cell_data[cindex].name;
							}

							println!("Cycling data comp");
							println!("Data name = {}", name);
							println!("Data comp = {}\n", rm.comp);
						}
						event::VirtualKeyCode::D =>
						{
							let name;
							let data_len = rm.m.point_data.len()
							             + rm.m. cell_data.len();

							rm.dindex = (rm.dindex + 1) % data_len;
							rm.comp = 0;

							// Cycle through point data first, then go to
							// cells if we're past the end of the points.
							if rm.dindex < rm.m.point_data.len()
							{
								rm.bind_point_data(display);
								name = &rm.m.point_data[rm.dindex].name;
							}
							else
							{
								// TODO: add a generic
								// rm.get_name() fn to handle this
								// index logic for both point and cell data

								let cindex = rm.dindex - rm.m.point_data.len();
								rm.bind_cell_data(display);
								name = &rm.m.cell_data[cindex].name;
							}

							println!("Cycling data array");
							println!("Data name = {}", name);
						}
						event::VirtualKeyCode::E =>
						{
							rm.edge_visibility = !rm.edge_visibility;
							println!("Toggling edge visibility {}",
								rm.edge_visibility);
						}
						event::VirtualKeyCode::M =>
						{
							println!("Cycling colormap");

							// Modulo wrapping happens inside
							// get_colormap().  Maybe I should make
							// bind_*_data() work like that too.
							s.map_index += 1;
							s.colormap = get_colormap(&mut s.map_index, display);
						}
						event::VirtualKeyCode::W =>
						{
							println!("Cycling warp");
							rm.warp_index += 1;
							rm.warp(display);
						}

						_ => {}
					}
				}
			},
			_ => return,
		},

		glutin::event::Event::NewEvents(cause) => match cause
		{
			glutin::event::StartCause::ResumeTimeReached {..} => (),
			glutin::event::StartCause::Init => (),
			_ => return,
		},
		_ => return,
	}

	// Apparently the rendering below must be in the same fn as the event
	// handling above, otherwise perf takes a massive hit

	let mut target = display.draw();

	s.display_diam = tnorm(target.get_dimensions());

	target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

	let fov: f32 = PI / 6.0;
	let zfar  = 1024.0;
	let znear = 0.1;

	let perspective =
			perspective_matrix(fov, zfar, znear, target.get_dimensions());

	// Light direction
	let light = [0.2, -0.6, -1.0f32];//[-1.4, -0.0, -0.7f32];
	//let light = [1.4, 0.4, -0.7f32];

	// Linear sampling works better than the default, especially around
	// texture 0
	let tex = glium::uniforms::Sampler::new(&s.colormap)
		.magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
		.minify_filter(glium::uniforms::MinifySamplerFilter::Linear);

	let bg_tex = glium::uniforms::Sampler::new(&s.bg.colormap)
		.magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
		.minify_filter(glium::uniforms::MinifySamplerFilter::Linear);

	let uniforms = uniform!
		{
			perspective: perspective,
			view : s.view ,
			world: s.world,
			model_mat: rm.mat,
			u_light: light,
			tex: tex,
			bg_tex: bg_tex,
		};

	let params = glium::DrawParameters
	{
		depth: glium::Depth
		{
			test: glium::draw_parameters::DepthTest::IfLessOrEqual,
			write: true,

			// High zoom levels are weird, but they're weirder without this.
			// Maybe increase depth buffer bits too?
			clamp: glium::draw_parameters::DepthClamp::Clamp,

			.. Default::default()
		},

		// Hack around z-fighting for edge display.  Units are pixels
		polygon_offset: glium::draw_parameters::PolygonOffset
		{
			factor: 1.01,
			//units: 3.0,
			//line: true,
			fill: true,
			.. Default::default()
		},

		// This is the default.  It could be increased, but the
		// polygon_offset works better than thickening.  Could expose to
		// user as an option
		line_width: Some(1.0),

		//backface_culling: glium::draw_parameters::BackfaceCullingMode
		//		::CullClockwise,

		.. Default::default()
	};

	target.draw(&s.bg.vertices, &s.bg.indices, &s.bg.program,
		&uniforms, &params).unwrap();

	// Clearing the depth again here forces the background to the back
	target.clear_depth(1.0);

	// TODO: move this to a RenderModel method?  Either pass program,
	// uniforms, and params as args or encapsulate them in RenderModel
	// struct.  Actually it seems nearly impossible to pass uniforms as
	// args.  I tried and failed to do so for the background.  Maybe
	// encapsulate them in another struct (state?) and pass that instead?
	target.draw((
		&rm.vertices,
		&rm.normals,
		&rm.scalar),
		&rm.indices,
		&s.face_program, &uniforms, &params).unwrap();

	if rm.edge_visibility
	{
		target.draw(
			&rm.edge_verts,
			&rm.edge_indices,
			&s.edge_program, &uniforms, &params).unwrap();
	}

	// TODO: draw axes, colormap legend

	// Swap buffers
	target.finish().unwrap();

	// TODO: take screenshot and compare for testing (just don't do it
	// everytime in the main loop.  maybe do that in a separate test/example)
}

//==============================================================================

