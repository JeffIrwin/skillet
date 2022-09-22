
//==============================================================================

// Standard
use std::io::Cursor;

//****************

// This crate
use skillet::*;
use skillet::consts::*;

//****************
// 3Ps
//****************

#[macro_use]
extern crate glium;

//****************

use vtkio::model::*;

//==============================================================================

fn main()
{
	println!();
	println!("{}:  Starting main()", ME);
	println!("{}:  {}", ME, mev!());
	println!();

	use glium::{glutin, Surface};

	let event_loop = glutin::event_loop::EventLoop::new();

	// include_bytes!() statically includes the file relative to this source path at compile time
	let icon = image::load(Cursor::new(&include_bytes!("../res/icon.png")),
			image::ImageFormat::Png).unwrap().to_rgba8();
	let winicon = Some(glutin::window::Icon::from_rgba(icon.to_vec(),
			icon.dimensions().0, icon.dimensions().1).unwrap());

	// TODO: maximize instead of hard-coding size
	let wb = glutin::window::WindowBuilder::new()
		.with_title(mev!())
		.with_window_icon(winicon)
		.with_inner_size(glutin::dpi::LogicalSize::new(1920.0, 1080.0));

	let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
	let display = glium::Display::new(wb, cb, &event_loop).unwrap();

	//=========================================================

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

	//// Black-Body Radiation.  TODO: this probably needs to be interpolated and expanded
	//let cmap = vec!
	//	[
	//		  0u8,   0u8,   0u8, 255u8,
	//		230u8,   0u8,   0u8, 255u8,
	//		230u8, 230u8,   0u8, 255u8,
	//		255u8, 255u8, 255u8, 255u8,
	//	];

	//=========================================================

	let image = glium::texture::RawImage1d::from_raw_rgba(cmap);
	let texture = glium::texture::SrgbTexture1d::new(&display, image).unwrap();

	//println!("image.w()   = {}", image.width);
	//println!("image.len() = {}", image.data.len());

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
		color: [f32; NM],
	}
	implement_vertex!(Node2, position2, color);

	// Even vectors and tensors will be rendered as "scalars", since you can only colormap one
	// component (or magnitude) at a time, which is a scalar
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

	// Split position and texture coordinates into separate arrays.  That way we can change texture
	// coordinates (e.g. rescale a colorbar range or load a different result) without sending the
	// position arrays to the GPU again

	//****************

	// TODO: cmd arg for VTK filename

	// VTK polydata files (or other piece types) can be saved as UnstructuredGrid (.vtu) in
	// ParaView with Filters -> Alphabetical -> Append datasets, in the mean time until I implement
	// polydata natively here

	use std::path::PathBuf;
	let file_path = PathBuf::from("./res/teapot.vtu");
	//let file_path = PathBuf::from("./res/ico64.vtu");
	//let file_path = PathBuf::from("./res/ico.vtu");

	//let file_path = PathBuf::from("./scratch/rbc-sinx.vtu");

	//// Legacy doesn't work?
	//let file_path = PathBuf::from("./scratch/teapot.vtk");
	//let file_path = PathBuf::from("./scratch/teapot-ascii.vtk");
	//let file_path = PathBuf::from("./scratch/cube.vtk");

	//let file_path = PathBuf::from("./scratch/fran_cut.vtk"); // polydata with texture coords
	//let file_path = PathBuf::from("./scratch/a.vtu");

	//let vtk = Vtk::parse_legacy_be(&file_path).expect(&format!("Failed to load file: {:?}", file_path));
	let vtk = Vtk::import(&file_path).expect(&format!("Failed to load file: {:?}", file_path));

	//let file_out = PathBuf::from("./scratch/ascii.vtu");
	//vtk.export_ascii(&file_out)
	//	.expect(&format!("Failed to save file: {:?}", file_out));
	//return;

	// TODO: match UnstructuredGrid vs PolyData, etc.
	let pieces = if let DataSet::UnstructuredGrid { pieces, ..} = vtk.data
	{
		pieces
	}
	else
	{
		panic!("UnstructuredGrid not found.  Wrong vtk data type");
	};

	println!("Number of pieces = {}", pieces.len());

	if pieces.len() > 1
	{
		// To do?  Render each piece as if it's a totally separate VTK file.  They could have
		// unrelated numbers of points, number and type of results, etc.
		unimplemented!("multiple pieces");
	}

	let piece = pieces[0].load_piece_data(None).unwrap();

	println!("Number of points = {}", piece.num_points());
	println!("Number of cells  = {}", piece.cells.types.len());
	println!();

	let points = piece.points.cast_into::<f32>().unwrap();

	//println!("points = {:?}", points);
	//println!();

	// Convert legacy into XML so we don't have to match conditionally
	let cells = piece.cells.cell_verts.into_xml();

	//println!("connectivity = {:?}", cells.0);
	//println!("types        = {:?}", piece.cells.types);
	//println!("offsets      = {:?}", cells.1);
	//println!();

	//println!("point 0 = {:?}", piece.data.point[0]);
	//println!();

	//// TODO: iterate attributes like this to get all pointdata (and cell data)
	//for a in &piece.data.point
	//{
	//	println!("a = {:?}", a);
	//}

	// Get the contents of the first pointdata array, assumining it's a scalar.  This is based on
	// write_attrib() from vtkio/src/writer.rs
	let pdata = match &piece.data.point[0]
	{
		Attribute::DataArray(DataArray {elem, data, ..}) =>
		{
			match elem
			{
				ElementType::Scalars{..}
				=>
				{
					// Cast everything to f32
					data.clone().cast_into::<f32>().unwrap()
				}

				// TODO: vectors, tensors
				_ => todo!()
			}
		}
		Attribute::Field {..} => unimplemented!("field attribute for point data")
	};

	//println!("pdata = {:?}", pdata);

	//****************

	// Get min/max of scalar.  This may not handle NaN correctly
	let (smin, smax) = get_bounds(&pdata);

	// Get point bounds
	let (xmin, xmax) = get_bounds(&(points.iter().skip(0).step_by(ND).copied().collect::<Vec<f32>>()));
	let (ymin, ymax) = get_bounds(&(points.iter().skip(1).step_by(ND).copied().collect::<Vec<f32>>()));
	let (zmin, zmax) = get_bounds(&(points.iter().skip(2).step_by(ND).copied().collect::<Vec<f32>>()));

	let xc = 0.5 * (xmin + xmax);
	let yc = 0.5 * (ymin + ymax);
	let zc = 0.5 * (zmin + zmax);

	let mut cen = vec![xc, yc, zc];

	let diam = norm(&sub(&[xmax, ymax, zmax], &[xmin, ymin, zmin]));
	println!("diam = {}", diam);

	println!("x in [{}, {}]", ff32(xmin), ff32(xmax));
	println!("y in [{}, {}]", ff32(ymin), ff32(ymax));
	println!("z in [{}, {}]", ff32(zmin), ff32(zmax));
	println!();

	// Capacity could be set ahead of time for tris with an extra pass over cell types to count
	// triangles
	let mut tris = Vec::new();
	for i in 0 .. piece.cells.types.len()
	{
		if piece.cells.types[i] == CellType::Triangle
		{
			// In vtkio, cells.0 is the actual connectivity, and cells.1 is the offset
			tris.push(cells.0[ (cells.1[i as usize] - 3) as usize ] as u32 );
			tris.push(cells.0[ (cells.1[i as usize] - 2) as usize ] as u32 );
			tris.push(cells.0[ (cells.1[i as usize] - 1) as usize ] as u32 );
		}
	}
	//println!("tris = {:?}", tris);

	// TODO: push other cell types to other buffers.  Draw them with separate calls to
	// target.draw().  Since vertices are duplicated per cell, there need to be parallel vertex and
	// scalar arrays too.  We could just push every cell type to a big list of tris, but that
    // wouldn't allow correct edge display or advanced filters that treat data at the cell level.

	let mut nodes   = Vec::with_capacity(tris.len());
	let mut scalar  = Vec::with_capacity(tris.len());
	let mut normals = Vec::with_capacity(tris.len());
	for i in 0 .. tris.len() / ND
	{
		let mut p: [f32; ND*ND] = [0.0; ND*ND];

		for j in 0 .. ND
		{
			p[ND*j + 0] = points[ND*tris[ND*i + j] as usize + 0];
			p[ND*j + 1] = points[ND*tris[ND*i + j] as usize + 1];
			p[ND*j + 2] = points[ND*tris[ND*i + j] as usize + 2];

			nodes.push(Node{position:
				[
					p[ND*j + 0],
					p[ND*j + 1],
					p[ND*j + 2],
				]});

			scalar.push(Scalar{tex_coord: ((pdata[tris[ND*i + j] as usize] - smin) / (smax - smin)) as f32 });
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

	let     tri_vbuf = glium::VertexBuffer::new(&display, &nodes).unwrap();
	let     tri_nbuf = glium::VertexBuffer::new(&display, &normals).unwrap();
	let     tri_ibuf = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
	let mut tri_sbuf = glium::VertexBuffer::new(&display, &scalar).unwrap();

	let vertex_shader_src = r#"
		#version 150

		in vec3 position;
		in vec3 normal;
		in float tex_coord;

		out vec3 v_normal;
		out vec3 v_position;
		out float n_tex_coord;

		uniform mat4 perspective;
		uniform mat4 view;
		uniform mat4 model;
		uniform mat4 world;

		void main()
		{
			n_tex_coord = tex_coord;
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
		in float n_tex_coord;

		out vec4 color;

		uniform vec3 u_light;
		uniform sampler1D tex;

		// Some of these parameters, like specular color or shininess, could be moved into
		// uniforms, or they're probably fine as defaults

		//const vec4 ambient_color = vec4(0.2, 0.0, 0.0, 1.0);
		//const vec4 diffuse_color = vec4(0.6, 0.0, 0.0, 1.0);
		//const vec4 specular_color = vec4(1.0, 1.0, 1.0, 1.0);
		const vec4 specular_color = vec4(0.1, 0.1, 0.1, 1.0);

		vec4 diffuse_color = texture(tex, n_tex_coord);
		vec4 ambient_color = diffuse_color * 0.1;

		void main()
		{
			float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);

			vec3 camera_dir = normalize(-v_position);
			vec3 half_direction = normalize(normalize(u_light) + camera_dir);
			float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 40.0);

			//color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
			color = ambient_color + diffuse * diffuse_color + specular * specular_color;
		}
	"#;

	// Background shader
	let bg_vertex_shader_src = r#"
		#version 150

		in vec2 position2;
		in vec4 color;
		out vec4 v_color;

		void main() {
			v_color = color;
			gl_Position = vec4(position2, 0, 1.0);
		}
	"#;

	let bg_fragment_shader_src = r#"
		#version 150

		in vec2 v_tex_coords;
		in vec4 v_color;
		out vec4 color;

		void main() {
			color = v_color;
		}
	"#;

	// background vertices
	let bg_verts = vec!
		[
			// Colors are based on jekyll cayman theme.  Something funny is going on, maybe gamma
			// correction
			Node2 { position2: [-1.0, -1.0], color: [0.03, 0.235, 0.235, 1.0] },
			Node2 { position2: [ 1.0, -1.0], color: [0.03, 0.350, 0.120, 1.0] },
			Node2 { position2: [ 1.0,  1.0], color: [0.03, 0.235, 0.235, 1.0] },
			Node2 { position2: [-1.0,  1.0], color: [0.03, 0.120, 0.350, 1.0] },
		];

	let bg_tri_vbuf = glium::VertexBuffer::new(&display, &bg_verts).unwrap();

	// No dupe
	let bg_tri_ibuf = glium::IndexBuffer::new(&display,
		glium::index::PrimitiveType::TrianglesList,
		&[
			0, 1, 2,
			2, 3, 0 as u32
		]).unwrap();

	let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src,
			None).unwrap();

	let bg_program = glium::Program::from_source(&display, bg_vertex_shader_src, bg_fragment_shader_src,
			None).unwrap();

	// Don't scale or translate here.  Model should always be identity unless I add an option
	// for a user to move one model relative to others
	let model = identity_matrix();

	// This is where transformations happen
	let mut world = identity_matrix();

	let fov: f32 = 3.141592 / 6.0;
	let zfar  = 1024.0;
	let znear = 0.1;

	// View must be initialized like this, because subsequent rotations are performed about its
	// fixed coordinate system.  Set eye from model bounds.  You could do some trig here on fov to
	// guarantee whole model is in view, but it's pretty close as is except for possible extreme
	// cases
	let mut eye = [0.0, 0.0, zmax + diam];
	let dir = [0.0, 0.0, -1.0];
	let up  = [0.0, 1.0,  0.0];
	let mut view = view_matrix(&eye, &dir, &up);

	let eye0 = eye;

	// Mouse buttons
	let mut lmb = false;
	let mut mmb = false;
	//let mut rmb = false;

	// Mouse position
	let mut x0 = 0.0;
	let mut y0 = 0.0;

	// Mouse scroll position.  Int is safer than float IMO.  The scroll wheel deltas are always +1
	// or -1 (or 2, 3, ... if you scroll fast).  For a very large float, adding 1.0 won't change
	// its value!  Ints won't have that problem, although they may overflow if you scroll for eons
	let mut z0: i64 = 0;

	// This initial value doesn't matter.  It will get set correctly after the first frame
	let mut display_diam = 1920.0;

	// Initial pan to center
	world = translate_matrix(&world, &neg(&cen));
	cen = vec![0.0; ND];

	println!("{}:  Starting main loop", ME);
	println!();
	event_loop.run(move |event, _, control_flow|
	{
		let next_frame_time = std::time::Instant::now() +
			std::time::Duration::from_nanos(16_666_667);
		*control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

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
						glium::glutin::event::MouseButton::Left =>
						{
							lmb = state == glium::glutin::event::ElementState::Pressed;
						},
						glium::glutin::event::MouseButton::Right =>
						{
							//rmb = state == glium::glutin::event::ElementState::Pressed;
						},
						glium::glutin::event::MouseButton::Middle =>
						{
							mmb = state == glium::glutin::event::ElementState::Pressed;
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

						// Right-hand normal to drag direction
						let mut u = [-(y - y0), -(x - x0), 0.0];

						let norm = norm(&u);
						u[0] /= norm;
						u[1] /= norm;
						// z is zero, no need to normalize

						let sensitivity = 0.0025;
						let theta = sensitivity * norm;

						// Push translation to model center, apply rotation, then pop trans
						world = translate_matrix(&world, &neg(&cen));
						world = rotate_matrix   (&world, &u, theta);
						world = translate_matrix(&world, &cen);

					}
					else if mmb
					{
						// Pan

						//println!("mmb drag");

						// TODO: scale sensitivity by zoom scale
						let sensitivity = 1.0 * diam / display_diam;
						let dx =  sensitivity * (x - x0);
						let dy = -sensitivity * (y - y0);

						let tran = [dx, dy, 0.0];

						world = translate_matrix(&world, &tran);

						// Panning moves rotation center too
						cen = add(&cen, &tran);
					}

					x0 = x;
					y0 = y;
				},
				glutin::event::WindowEvent::MouseWheel {delta, ..} =>
				{
					//println!("delta = {:?}", delta);
					//println!("z0 = {}", z0);

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

					z0 += dz as i64;

					// This sign convention matches ParaView, although the opposite scroll/zoom
					// convention does exist

					let sensitivity = 0.1;
					eye[2] = eye0[2] - sensitivity * diam * z0 as f32;

					// TODO: ParaView actually has two ways to "zoom": the RMB-drag moves the eye
					// of the view, like here, while the scroll wheel scales the world, which
					// I still have to do.  Implement the scaling method and patch it in here.
					// Move this code to the rmb drag match-case.

					view = view_matrix(&eye, &dir, &up);
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

		let perspective = perspective_matrix(fov, zfar, znear, target.get_dimensions());

		// Light direction
		let light = [0.2, -0.6, -1.0f32];//[-1.4, -0.0, -0.7f32];
		//let light = [1.4, 0.4, -0.7f32];

		// Linear sampling works better than the default, especially around texture 0
		let tex = glium::uniforms::Sampler::new(&texture)
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
			};

		let params = glium::DrawParameters
		{
			depth: glium::Depth
			{
				test: glium::draw_parameters::DepthTest::IfLess,
				write: true,
				.. Default::default()
			},
			//backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
			.. Default::default()
		};

		target.draw(&bg_tri_vbuf, &bg_tri_ibuf, &bg_program,
			&uniforms, &params).unwrap();

		// Clearing the depth again here forces the background to the back
		target.clear_depth(1.0);

		target.draw((&tri_vbuf, &tri_nbuf, &tri_sbuf), &tri_ibuf, &program,
			&uniforms, &params).unwrap();

		// Swap buffers
		target.finish().unwrap();

		// TODO: take screenshot and compare for testing
	});
}

//==============================================================================

fn ff32(num: f32) -> String
{
	// Rust doesn't do optional args, so we have 2 fn's instead
	fmt_f32(num, 12, 5, 2)
}

fn fmt_f32(num: f32, width: usize, precision: usize, exp_pad: usize) -> String
{
	// https://stackoverflow.com/questions/65264069/alignment-of-floating-point-numbers-printed-in-scientific-notation

	let mut num = format!("{:.precision$e}", num, precision = precision);

	// Safe to `unwrap` as `num` is guaranteed to contain `'e'`
	let exp = num.split_off(num.find('e').unwrap());

	let (sign, exp) = if exp.starts_with("e-")
	{
		('-', &exp[2..])
	}
	else
	{
		('+', &exp[1..])
	};
	num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

	format!("{:>width$}", num, width = width)
}

//==============================================================================

fn get_bounds(x: &[f32]) -> (f32, f32)
{
	let mut xmin = x[0];
	let mut xmax = x[0];
	for i in 1 .. x.len()
	{
		if x[i] < xmin { xmin = x[i]; }
		if x[i] > xmax { xmax = x[i]; }
	}
	(xmin, xmax)
}

//==============================================================================

fn rotate_matrix(m: &[[f32; NM]; NM], u: &[f32; ND], theta: f32) -> [[f32; NM]; NM]
{
	// General axis-angle rotation about an axis vector [x,y,z] by angle theta.  Vector must be
	// normalized!  Apply rotation r to input matrix m and return m * r.

	// Skip identity/singular case. Caller likely set vector to garbage
	if theta == 0.0f32
	{
		return *m;
	}

	// Ref:
	//
	//     https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
	//
	// c.f.
	//
	//     https://github.com/JeffIrwin/temple-viewer/blob/c25f28cf3457edc136d213fd01df47d826bc279b/Math.HC#L39

	let c = theta.cos();
	let s = theta.sin();
	let t = 1.0 - c;

	let x = u[0];
	let y = u[1];
	let z = u[2];

	let r =
		[
			[c + x*x*t  ,  x*y*t - z*s,  z*x*t + y*s, 0.0],
			[x*y*t + z*s,    c + y*y*t,  y*z*t - x*s, 0.0],
			[z*x*t - y*s,  y*z*t + x*s,    c + z*z*t, 0.0],
			[        0.0,          0.0,          0.0, 1.0],
		];

	//println!("theta = {:?}", theta);
	//println!("r = {:?}", r);

	mul_mat4(m, &r)
}

//==============================================================================

fn translate_matrix(m: &[[f32; NM]; NM], u: &[f32]) -> [[f32; NM]; NM]
{
	// Translate in place and apply after m

	let t =
		[
			[ 1.0,  0.0,  0.0, 0.0],
			[ 0.0,  1.0,  0.0, 0.0],
			[ 0.0,  0.0,  1.0, 0.0],
			[u[0], u[1], u[2], 1.0],
		];

	mul_mat4(m, &t)
}

//==============================================================================

fn mul_mat4(a: &[[f32; NM]; NM], b: &[[f32; NM]; NM]) -> [[f32; NM]; NM]
{
	let mut c = [[0.0; NM]; NM];
	for i in 0 .. NM
	{
		for j in 0 .. NM
		{
			for k in 0 .. NM
			{
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	c
}

//// It's called an arg because I shout "arg!" at rustc when it complains about mismatches between
//// array vs slice vs Vec
//fn mul_matrix(a: &[&[f32]], b: &[&[f32]]) -> Vec<Vec<f32>>
//{
//	if a[0].len() != b.len()
//	{
//		panic!("Inner dimensions must agree for mul_matrix()");
//	}
//	// check dims > 0
//
//	//let mut c = [[0.0; a.len()]; b[0].len()];
//	let mut c = vec![vec![0.0; a.len()]; b[0].len()];
//
//	for i in 0 .. a.len()
//	{
//		for j in 0 .. b[0].len()
//		{
//			for k in 0 .. b.len()
//			{
//				c[i][j] += a[i][k] * b[k][j];
//			}
//		}
//	}
//	c
//}

fn identity_matrix() -> [[f32; NM]; NM]
{[
	[1.0, 0.0, 0.0, 0.0],
	[0.0, 1.0, 0.0, 0.0],
	[0.0, 0.0, 1.0, 0.0],
	[0.0, 0.0, 0.0, 1.0],
]}

//==============================================================================

// TODO: consider using nalgebra crate for vector/matrix wrapper types with operator overloading

// Can't overload "+" operator because rust makes it impossible by design without a wrapper type :(
fn add(a: &[f32], b: &[f32]) -> Vec<f32>
{
	if a.len() != b.len()
	{
		panic!("Incorrect length for add() arguments");
	}

	//a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()

	let mut c = Vec::with_capacity(a.len());
	for i in 0 .. a.len()
	{
		c.push(a[i] + b[i]);
	}
	c
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32>
{
	if a.len() != b.len()
	{
		panic!("Incorrect length for sub() arguments");
	}

	//a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()

	let mut c = Vec::with_capacity(a.len());
	for i in 0 .. a.len()
	{
		c.push(a[i] - b[i]);
	}
	c
}

fn neg(a: &[f32]) -> Vec<f32>
{
	a.iter().map(|x| -x).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32
{
	if a.len() != b.len()
	{
		panic!("Incorrect length for dot() arguments");
	}

	//// Unreadable IMO
	//a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()

	let mut d = 0.0;
	for i in 0 .. a.len()
	{
		d += a[i] * b[i];
	}
	d
}

fn norm(a: &[f32]) -> f32
{
	dot(&a, &a).sqrt()
}

fn tnorm((w, h): (u32, u32)) -> f32
{
	// Tuple norm
	((w*w) as f32 + (h*h) as f32).sqrt()
}

fn normalize(a: &[f32]) -> Vec<f32>
{
	let norm = norm(a);
	a.iter().map(|x| x / norm).collect()
}

fn cross(a: &[f32], b: &[f32]) -> [f32; ND]
{
	if a.len() != ND || b.len() != ND
	{
		// 3D only.  This could return a Return value instead.  I can't put this check into the
		// function signature because then rust will only accept arrays as args, not slices or
		// Vecs.
		panic!("Incorrect length for cross() argument.  Expected length {}", ND);
	}

	[
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	]
}

//==============================================================================

fn view_matrix(position: &[f32; ND], direction: &[f32; ND], up: &[f32; ND]) -> [[f32; NM]; NM]
{
	// Ref:
	//
	//     https://github.com/JeffIrwin/glfw/blob/9416a43404934cc54136e988a233bee64d4d48fb/deps/linmath.h#L404
	//
	// Note this version has a "direction" arg instead of "center", but it's equivalent

	let f = normalize(direction);
	let s = normalize(&cross(&f, up));
	let u = cross(&s, &f);

	let p = [-dot(position, &s),
			 -dot(position, &u),
			 -dot(position, &f)];

	[
		[s[0], u[0], -f[0], 0.0],
		[s[1], u[1], -f[1], 0.0],
		[s[2], u[2], -f[2], 0.0],
		[p[0], p[1], -p[2], 1.0],
	]
}

//==============================================================================

fn perspective_matrix(fov: f32 , zfar: f32, znear: f32, (width, height): (u32, u32))
		-> [[f32; NM]; NM]
{
	// Right-handed (as god intended)
	//
	// Ref:  http://perry.cz/articles/ProjectionMatrix.xhtml

	let aspect_ratio = height as f32 / width as f32;
	let f = 1.0 / (fov / 2.0).tan();

	[
		[f * aspect_ratio, 0.0, 0.0                           ,  0.0],
		[         0.0    ,   f, 0.0                           ,  0.0],
		[         0.0    , 0.0, -    (zfar+znear)/(zfar-znear), -1.0],
		[         0.0    , 0.0, -(2.0*zfar*znear)/(zfar-znear),  0.0],
	]
}

//==============================================================================

