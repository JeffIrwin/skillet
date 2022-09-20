
//==============================================================================

#[macro_use]
extern crate glium;

//****************

use vtkio::model::*;

//****************

// Global constants

// Number of dimensions
const ND: usize = 3;

// Augmented matrix size
const NM: usize = 4;

//==============================================================================

fn main() {
	println!("skillet:  Starting main()");

	use glium::{glutin, Surface};

	let event_loop = glutin::event_loop::EventLoop::new();
	let wb = glutin::window::WindowBuilder::new();
	let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
	let display = glium::Display::new(wb, cb, &event_loop).unwrap();

	//=========================================================

	// Define the colormap.  Hard-code for now

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

	println!("image.w()   = {}", image.width);
	println!("image.len() = {}", image.data.len());

	let texture = glium::texture::SrgbTexture1d::new(&display, image).unwrap();

	#[derive(Copy, Clone, Debug)]
	struct Node {
		position: [f32; ND]
	}
	implement_vertex!(Node, position);

	// Even vectors and tensors will be rendered as "scalars", since you can only colormap one
	// component (or magnitude) at a time, which is a scalar
	#[derive(Copy, Clone)]
	struct Scalar {
		tex_coord: f32,
	}
	implement_vertex!(Scalar, tex_coord);

	#[derive(Copy, Clone, Debug)]
	struct Normal {
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
	//let file_path = PathBuf::from("./res/ico64.vtu");
	//let file_path = PathBuf::from("./res/ico.vtu");
	let file_path = PathBuf::from("./res/teapot.vtu");

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
	let pieces = if let DataSet::UnstructuredGrid { pieces, .. } = vtk.data {
		pieces
	} else {
		panic!("UnstructuredGrid not found.  Wrong vtk data type");
	};

	println!("n pieces = {}", pieces.len());

	let piece = pieces[0].load_piece_data(None).unwrap();

	println!("num_points = {}", piece.num_points());
	println!("num_cells  = {}", piece.cells.types.len());
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

	// Get the contents of the first pointdata array, assumining it's a scalar

	let pdata: Vec<f32> = match &piece.data.point[0] {
		Attribute::DataArray(DataArray { elem, data, .. }) => {
			match elem {
				ElementType::Scalars {
					..
				} => {

					// This is based on write_attrib() from vtkio/src/writer.rs

					// Cast everything to f32.  TODO: other types?
					data.clone().cast_into::<f32>().unwrap()
				}

				// TODO: vectors, tensors
				_ => todo!()

			}
		}
		Attribute::Field {..} => todo!()
	};

	//println!("pdata = {:?}", pdata);

	//****************

	// Get min/max of scalar.  This may not handle NaN correctly
	let mut smin = pdata[0];
	let mut smax = pdata[0];
	for i in 1 .. pdata.len()
	{
		if pdata[i] < smin { smin = pdata[i]; }
		if pdata[i] > smax { smax = pdata[i]; }
	}

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
	// scalar arrays too

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
				]});
		}
	}

	println!("node   0 = {:?}", nodes[0]);
	println!("node   1 = {:?}", nodes[1]);
	println!("node   2 = {:?}", nodes[2]);

	println!("normal 0 = {:?}", normals[0]);

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

		void main() {
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

		//const vec4 ambient_color = vec4(0.2, 0.0, 0.0, 1.0);
		//const vec4 diffuse_color = vec4(0.6, 0.0, 0.0, 1.0);
		//const vec4 specular_color = vec4(1.0, 1.0, 1.0, 1.0);
		const vec4 specular_color = vec4(0.1, 0.1, 0.1, 1.0);

		vec4 diffuse_color = texture(tex, n_tex_coord);
		vec4 ambient_color = diffuse_color * 0.1;

		void main() {
			float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);

			vec3 camera_dir = normalize(-v_position);
			vec3 half_direction = normalize(normalize(u_light) + camera_dir);
			float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 40.0);

			//color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
			color = ambient_color + diffuse * diffuse_color + specular * specular_color;
		}
	"#;

	let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src,
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
	// fixed coordinate system
	let eye = [0.0, 0.0, 30.0];
	let dir = [0.0, 0.0, -1.0];
	let up  = [0.0, 1.0,  0.0];
	let view = view_matrix(&eye, &dir, &up);

	// Mouse buttons
	let mut lmb = false;
	//let mut rmb = false; // TODO: panning, zooming
	//let mut mmb = false;

	// Mouse position
	let mut x0 = 0.0;
	let mut y0 = 0.0;

	event_loop.run(move |event, _, control_flow| {
		let next_frame_time = std::time::Instant::now() +
			std::time::Duration::from_nanos(16_666_667);
		*control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

		match event
		{
			glutin::event::Event::WindowEvent { event, .. } => match event
			{
				glutin::event::WindowEvent::CloseRequested =>
				{
					*control_flow = glutin::event_loop::ControlFlow::Exit;
					return;
				},
				glutin::event::WindowEvent::MouseInput  {state, button, .. } =>
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
							//mmb = state == glium::glutin::event::ElementState::Pressed;
						},
						_ => ()
					}

				},
				glutin::event::WindowEvent::CursorMoved {position, .. } => {

					let x = position.x as f32;
					let y = position.y as f32;

					if lmb
					{
						//println!("position = {:?}", position);

						// Right-hand normal to drag direction
						let mut u = [-(y - y0), -(x - x0), 0.0f32];

						let norm = norm(&u);
						u[0] /= norm;
						u[1] /= norm;
						// z is zero, no need to normalize

						let sensitivity = 0.005;
						let theta = sensitivity * norm;
						world = rotate_matrix(&world, &u, theta);
					}

					x0 = x;
					y0 = y;
				},
				_ => return,
			},

			glutin::event::Event::NewEvents(cause) => match cause
			{
				glutin::event::StartCause::ResumeTimeReached { .. } => (),
				glutin::event::StartCause::Init => (),
				_ => return,
			},
			_ => return,
		}

		let mut target = display.draw();

		// TODO: gradient bg
		target.clear_color_and_depth((0.322, 0.341, 0.431, 1.0), 1.0);

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

		let params = glium::DrawParameters {
			depth: glium::Depth {
				test: glium::draw_parameters::DepthTest::IfLess,
				write: true,
				.. Default::default()
			},
			backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
			.. Default::default()
		};

		target.draw((&tri_vbuf, &tri_nbuf, &tri_sbuf), &tri_ibuf, &program,
			&uniforms, &params).unwrap();

		target.finish().unwrap();
	});
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

	mat4x4_mul(m, &r)
}

//==============================================================================

// TODO: make a general slice version returning Vec for any size a, b (panic if inner dims don't
// agree)
fn mat4x4_mul(a: &[[f32; NM]; NM], b: &[[f32; NM]; NM]) -> [[f32; NM]; NM]
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

fn identity_matrix() -> [[f32; NM]; NM]
{[
	[1.0, 0.0, 0.0, 0.0],
	[0.0, 1.0, 0.0, 0.0],
	[0.0, 0.0, 1.0, 0.0],
	[0.0, 0.0, 0.0, 1.0]
]}

//==============================================================================

// TODO: consider using nalgebra crate for vector/matrix wrapper types with operator overloading

// Can't overload "-" operator because rust makes it impossible by design without a wrapper type :(
fn sub(a: &[f32], b: &[f32]) -> Vec<f32>
{
	if a.len() != b.len()
	{
		panic!("Incorrect length for dot() arguments");
	}

	//a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()

	let mut c = Vec::with_capacity(a.len());
	for i in 0 .. a.len()
	{
		c.push(a[i] - b[i]);
	}
	c
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

