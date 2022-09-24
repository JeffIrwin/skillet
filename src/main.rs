
//==============================================================================

// Standard
use std::env;
use std::io::Cursor;

//****************

// This crate
use skillet::*;
use crate::colormaps::*;
use crate::consts::*;
use crate::math::*;
use crate::model::*;
use crate::utils::*;

//****************
// 3P(s)
//****************

#[macro_use]
extern crate glium;

//==============================================================================

fn main()
{
	use glium::{glutin, glutin::event_loop::EventLoop, Surface};
	use std::path::PathBuf;

	println!();
	println!("{}:  Starting main()", ME);
	println!("{}:  {}", ME, mev!());
	println!();

	//let exe = env::current_exe().unwrap();
	//let exe_base = exe.file_name().unwrap().to_str().unwrap();

	let exe = env::current_exe().unwrap();
	let exe_base_opt = exe.file_name().unwrap().to_str();
	let exe_base = match exe_base_opt
	{
			Some(inner) => inner,
			None        => "skillet.exe",
	};

	//let exe_base =
	//{
	//	let exe = env::current_exe().unwrap();
	//	let exe_base_opt = exe.file_name().unwrap().to_str();
	//	match exe_base_opt
	//	{
	//			Some(inner) => inner,
	//			None        => "skillet.exe",
	//	}
	//};

	////let exe_base = exe.file_name().display();
	////let exe_base = exe.file_name().unwrap().to_os_string().into_string().unwrap();
	//let exe_base = match
	//	{
	//		let exe = env::current_exe().unwrap();
	//		let exef = exe.file_name().unwrap();
	//		exef.to_str()
	//	}
	//	{
	//		Some(inner) => inner,
	//		None        => "skillet.exe",
	//	};

	// Switch to clap if args become more complicated than a single filename
	let args: Vec<String> = env::args().collect();
	if args.len() < 2
	{
		println!("Error: bad cmd args");
		println!("Usage:");
		println!();
		println!("    {} FILE.VTK", exe_base);
		println!();
		return;
	}

	let file_path = PathBuf::from(args[1].clone());

	let m = import(file_path);

	let event_loop = EventLoop::new();

	// include_bytes!() statically includes the file relative to this source
	// path at compile time

	// TODO: cleanup some of these use paths
	let icon = image::load(Cursor::new(&include_bytes!("../res/icon.png")),
			image::ImageFormat::Png).unwrap().to_rgba8();
	let winicon = Some(glutin::window::Icon::from_rgba(icon.to_vec(),
			icon.dimensions().0, icon.dimensions().1).unwrap());

	let wb = glutin::window::WindowBuilder::new()
		.with_title(mev!())
		.with_window_icon(winicon)
		.with_maximized(true);

	let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
	let display = glium::Display::new(wb, cb, &event_loop).unwrap();

	let colormap = get_colormap(&display);
	let bg_colormap = get_bg_colormap(&display);

	#[derive(Copy, Clone, Debug)]
	struct Node
	{
		// 3D node
		position: [f32; ND]
	}
	implement_vertex!(Node, position);

	#[derive(Copy, Clone, Debug)]
	struct Node2
	{
		// 2D node for background
		position2: [f32; N2],
		tex_coord: f32,
	}
	implement_vertex!(Node2, position2, tex_coord);

	// Even vectors and tensors will be rendered as "scalars", since you can
	// only colormap one component (or magnitude) at a time, which is a scalar
	#[derive(Copy, Clone)]
	struct Scalar
	{
		tex_coord: f32,
	}
	implement_vertex!(Scalar, tex_coord);

	#[derive(Copy, Clone, Debug)]
	struct Normal
	{
		normal: [f32; ND]
	}
	implement_vertex!(Normal, normal);

	// Split position and texture coordinates into separate arrays.  That way we
	// can change texture coordinates (e.g. rescale a colorbar range or load
	// a different result) without sending the position arrays to the GPU again

	//****************

	// VTK polydata files (or other piece types) can be saved as
	// UnstructuredGrid (.vtu) in ParaView with Filters -> Alphabetical ->
	// Append datasets, in the mean time until I implement polydata natively
	// here

	//****************

	// Get min/max of scalar
	let (smin, smax) = get_bounds(&m.pdata);

	// Get point xyz bounds

	let (xmin, xmax) = get_bounds(&(m.points.iter().skip(0)
			.step_by(ND).copied().collect::<Vec<f32>>()));

	let (ymin, ymax) = get_bounds(&(m.points.iter().skip(1)
			.step_by(ND).copied().collect::<Vec<f32>>()));

	let (zmin, zmax) = get_bounds(&(m.points.iter().skip(2)
			.step_by(ND).copied().collect::<Vec<f32>>()));

	let xc = 0.5 * (xmin + xmax);
	let yc = 0.5 * (ymin + ymax);
	let zc = 0.5 * (zmin + zmax);

	let mut cen = vec![xc, yc, zc];

	let diam = norm(&sub(&[xmax, ymax, zmax], &[xmin, ymin, zmin]));

	//println!("diam = {}", diam);
	println!("x in [{}, {}]", ff32(xmin), ff32(xmax));
	println!("y in [{}, {}]", ff32(ymin), ff32(ymax));
	println!("z in [{}, {}]", ff32(zmin), ff32(zmax));
	println!();

	// Capacity could be set ahead of time for tris with an extra pass over cell
	// types to count triangles
	let mut tris = Vec::new();
	for i in 0 .. m.types.len()
	{
		if m.types[i] == vtkio::model::CellType::Triangle
		{
			tris.push(m.cells[ (m.offsets[i as usize] - 3) as usize ] as u32 );
			tris.push(m.cells[ (m.offsets[i as usize] - 2) as usize ] as u32 );
			tris.push(m.cells[ (m.offsets[i as usize] - 1) as usize ] as u32 );
		}
	}
	//println!("tris = {:?}", tris);

	// TODO: push other cell types to other buffers.  Draw them with separate
	// calls to target.draw().  Since vertices are duplicated per cell, there
	// need to be parallel vertex and scalar arrays too.  We could just push
	// every cell type to a big list of tris, but that wouldn't allow correct
	// edge display or advanced filters that treat data at the cell level.

	// TODO: split scalar handling to a separate loop (and eventually a separate
	// fn).  Mesh geometry will only be loaded once, but scalars may be
	// processed multiple times as the user cycles through results to display

	let mut nodes   = Vec::with_capacity(tris.len());
	let mut scalar  = Vec::with_capacity(tris.len());
	let mut normals = Vec::with_capacity(tris.len());
	for i in 0 .. tris.len() / ND
	{
		// Local array containing the coordinates of the vertices of a single
		// triangle
		let mut p: [f32; ND*ND] = [0.0; ND*ND];

		for j in 0 .. ND
		{
			p[ND*j + 0] = m.points[ND*tris[ND*i + j] as usize + 0];
			p[ND*j + 1] = m.points[ND*tris[ND*i + j] as usize + 1];
			p[ND*j + 2] = m.points[ND*tris[ND*i + j] as usize + 2];

			nodes.push(Node{position:
				[
					p[ND*j + 0],
					p[ND*j + 1],
					p[ND*j + 2],
				]});

			let s = m.pdata[tris[ND*i + j] as usize];
			scalar.push(Scalar{tex_coord: ((s-smin) / (smax-smin)) as f32 });
		}

		let p01 = sub(&p[3..6], &p[0..3]);
		let p02 = sub(&p[6..9], &p[0..3]);

		let nrm = normalize(&cross(&p01, &p02));

		// Use inward normal for RH coordinate system
		for _j in 0 .. ND
		{
			normals.push(Normal{normal:
				[
					-nrm[0],
					-nrm[1],
					-nrm[2],
				]
			});
		}
	}

	//println!("node   0 = {:?}", nodes[0]);
	//println!("node   1 = {:?}", nodes[1]);
	//println!("node   2 = {:?}", nodes[2]);
	//println!("normal 0 = {:?}", normals[0]);

	let     tri_vbuf = glium::VertexBuffer::new(&display, &nodes  ).unwrap();
	let     tri_nbuf = glium::VertexBuffer::new(&display, &normals).unwrap();
	let mut tri_sbuf = glium::VertexBuffer::new(&display, &scalar ).unwrap();

	let     tri_ibuf = glium::index::NoIndices(
			glium::index::PrimitiveType::TrianglesList);

	let vertex_shader_src = r#"
		#version 150

		in vec3 position;
		in vec3 normal;
		in float tex_coord;

		out vec3 v_normal;
		out vec3 v_position;
		out float v_tex_coord;

		uniform mat4 perspective;
		uniform mat4 view;
		uniform mat4 model;
		uniform mat4 world;

		void main()
		{
			v_tex_coord = tex_coord;
			mat4 modelview = view * world * model;
			v_normal = transpose(inverse(mat3(modelview))) * normal;
			gl_Position = perspective * modelview * vec4(position, 1.0);
			v_position = gl_Position.xyz / gl_Position.w;
		}
	"#;

	// TODO: Gouraud option

	// Blinn-Phong
	let fragment_shader_src = r#"
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
	"#;

	// Background shader
	let bg_vertex_shader_src = r#"
		#version 150

		in vec2 position2;
		in float tex_coord;
		out float v_tex_coord;

		void main() {
			v_tex_coord = tex_coord;
			gl_Position = vec4(position2, 0, 1.0);
		}
	"#;

	let bg_fragment_shader_src = r#"
		#version 150

		in float v_tex_coord;
		out vec4 color;

		uniform sampler1D bg_tex;

		void main() {
			color = texture(bg_tex, v_tex_coord);
		}
	"#;

	// background vertices
	let bg_verts = vec!
		[
			Node2 { position2: [-1.0, -1.0], tex_coord: 0.5, },
			Node2 { position2: [ 1.0, -1.0], tex_coord: 1.0, },
			Node2 { position2: [ 1.0,  1.0], tex_coord: 0.5, },
			Node2 { position2: [-1.0,  1.0], tex_coord: 0.0, },
		];

	let bg_tri_vbuf = glium::VertexBuffer::new(&display, &bg_verts).unwrap();

	// No dupe
	let bg_tri_ibuf = glium::IndexBuffer::new(&display,
		glium::index::PrimitiveType::TrianglesList,
		&[
			0, 1, 2,
			2, 3, 0 as u32
		]).unwrap();

	let program = glium::Program::from_source(&display, vertex_shader_src,
			fragment_shader_src, None).unwrap();

	let bg_program = glium::Program::from_source(&display, bg_vertex_shader_src,
			bg_fragment_shader_src, None).unwrap();

	// Don't scale or translate here.  Model matrix should always be identity
	// unless I add an option for a user to move one model relative to others
	let model = identity_matrix();

	// This is where transformations happen
	let mut world = identity_matrix();

	let fov: f32 = PI / 6.0;
	let zfar  = 1024.0;
	let znear = 0.1;

	// View must be initialized like this, because subsequent rotations are
	// performed about its fixed coordinate system.  Set eye from model bounds.
	// You could do some trig here on fov to guarantee whole model is in view,
	// but it's pretty close as is except for possible extreme cases

	let mut eye = [0.0, 0.0, zmax + diam];
	let dir = [0.0, 0.0, -1.0];
	let up  = [0.0, 1.0,  0.0];

	let mut view = view_matrix(&eye, &dir, &up);

	// Mouse button states
	let mut lmb = false;
	let mut mmb = false;
	let mut rmb = false;

	// Mouse position from last frame
	let mut x0 = 0.0;
	let mut y0 = 0.0;

	// Scroll wheel zoom factor
	let mut scale_cum = 1.0;

	// This initial value doesn't matter.  It will get set correctly after the
	// first frame
	let mut display_diam = 1920.0;

	// Initial pan to center
	world = translate_matrix(&world, &neg(&cen));
	cen = vec![0.0; ND];

	const PRESSED: glutin::event::ElementState
	             = glutin::event::ElementState::Pressed;

	println!("{}:  Starting main loop", ME);
	println!();
	event_loop.run(move |event, _, control_flow|
	{
		let next_frame_time = std::time::Instant::now() +
				std::time::Duration::from_nanos(16_666_667);
		*control_flow =
				glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

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
				glutin::event::WindowEvent::MouseInput  {state, button, ..} =>
				{
					//println!("state, button = {:?}, {:?}", state, button);

					match button
					{
						glutin::event::MouseButton::Left =>
						{
							lmb = state == PRESSED;
						},
						glutin::event::MouseButton::Right =>
						{
							rmb = state == PRESSED;
						},
						glutin::event::MouseButton::Middle =>
						{
							mmb = state == PRESSED;
						},
						_ => ()
					}

				},
				glutin::event::WindowEvent::CursorMoved {position, ..} =>
				{
					//println!("position = {:?}", position);

					let x = position.x as f32;
					let y = position.y as f32;

					if lmb
					{
						// Rotate about axis within the xy screen plane
						//
						// TODO: handle shift-lmb as z rotation

						// Right-hand normal to drag direction
						let mut u = [-(y - y0), -(x - x0), 0.0];

						let norm = norm(&u);
						u[0] /= norm;
						u[1] /= norm;
						// z is zero, no need to normalize

						let sensitivity = 0.0035;
						let theta = sensitivity * norm;

						// Push translation to model center, apply rotation,
						// then pop trans
						world = translate_matrix(&world, &neg(&cen));
						world = rotate_matrix   (&world, &u, theta);
						world = translate_matrix(&world, &cen);

					}
					else if mmb
					{
						// xy pan

						//println!("mmb drag");

						let sensitivity = 1.5 * diam //* scale_cum
								/ display_diam;

						let dx =  sensitivity * (x - x0);// / display_h;
						let dy = -sensitivity * (y - y0);// / display_w;

						let tran = [dx, dy, 0.0];

						world = translate_matrix(&world, &tran);

						// Panning moves rotation center too
						cen = add(&cen, &tran);
					}
					else if rmb
					{
						// z pan (eye motion zoom)
						//
						// This uses the opposite sign convention of ParaView,
						// but I think it feels more consistent with the scroll
						// wheel action:  scrolling up has a similar effect as
						// rmb dragging up

						let dz = y - y0;

						let sensitivity = 0.003;
						eye[2] += sensitivity * scale_cum * diam * dz;
						view = view_matrix(&eye, &dir, &up);
					}

					x0 = x;
					y0 = y;
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
							b
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
					scale_cum *= scale;

					//println!("scale = {}", scale);

					world = scale_matrix(&world, scale);
					cen = scale_vec(&cen, scale);
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

		let mut target = display.draw();

		display_diam = tnorm(target.get_dimensions());

		target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

		let perspective =
				perspective_matrix(fov, zfar, znear, target.get_dimensions());

		// Light direction
		let light = [0.2, -0.6, -1.0f32];//[-1.4, -0.0, -0.7f32];
		//let light = [1.4, 0.4, -0.7f32];

		// Linear sampling works better than the default, especially around
		// texture 0
		let tex = glium::uniforms::Sampler::new(&colormap)
			.magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
			.minify_filter(glium::uniforms::MinifySamplerFilter::Linear);

		let bg_tex = glium::uniforms::Sampler::new(&bg_colormap)
			.magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
			.minify_filter(glium::uniforms::MinifySamplerFilter::Linear);

		let uniforms = uniform!
			{
				perspective: perspective,
				view : view ,
				world: world,
				model: model,
				u_light: light,
				tex: tex,
				bg_tex: bg_tex,
			};

		let params = glium::DrawParameters
		{
			depth: glium::Depth
			{
				test: glium::draw_parameters::DepthTest::IfLess,
				write: true,
				.. Default::default()
			},
			//backface_culling: glium::draw_parameters::BackfaceCullingMode
			//		::CullClockwise,
			.. Default::default()
		};

		target.draw(&bg_tri_vbuf, &bg_tri_ibuf, &bg_program,
			&uniforms, &params).unwrap();

		// Clearing the depth again here forces the background to the back
		target.clear_depth(1.0);

		target.draw((&tri_vbuf, &tri_nbuf, &tri_sbuf), &tri_ibuf, &program,
			&uniforms, &params).unwrap();

		// TODO: draw axes, colormap legend

		// Swap buffers
		target.finish().unwrap();

		// TODO: take screenshot and compare for testing (just don't do it
		// everytime in the main loop)
	});
}

//==============================================================================

