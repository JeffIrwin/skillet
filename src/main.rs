
//==============================================================================

//use std::cmp;
//use std::io::Cursor;

//****************

#[macro_use]
extern crate glium;

//****************

use vtkio::model::*;

//****************

// Global constants

// Number of dimensions
const ND: usize = 3;

//****************

// TODO: glfw, or figure out key/mouse handling with glutin

//==============================================================================

fn main() {
    println!("skillet:  Starting main()");

    #[allow(unused_imports)]
    use glium::{glutin, Surface};

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new();
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
    //    [
    //          0u8,   0u8,   0u8, 255u8,
    //        230u8,   0u8,   0u8, 255u8,
    //        230u8, 230u8,   0u8, 255u8,
    //        255u8, 255u8, 255u8, 255u8,
    //    ];

    //=========================================================

    let image = glium::texture::RawImage1d::from_raw_rgba(cmap);

    println!("image.w()   = {}", image.width);
    println!("image.len() = {}", image.data.len());

    let texture = glium::texture::SrgbTexture1d::new(&display, image).unwrap();

    #[derive(Copy, Clone)]
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
    //    .expect(&format!("Failed to save file: {:?}", file_out));
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

    // TODO: is it right to cast into vec here?  Can we defer and only clone it to vertex buffer
    // "nodes"?
    let points = piece.points.into_vec::<f32>().unwrap();

    //println!("points = {:?}", points);
    //println!();

    // Convert legacy into XML so we don't have to match conditionally
    let vcells = piece.cells.cell_verts.into_xml();

    //println!("connectivity = {:?}", vcells.0);
    //println!("types        = {:?}", piece.cells.types);
    //println!("offsets      = {:?}", vcells.1);
    //println!();

    //println!("point 0 = {:?}", piece.data.point[0]);
    //println!();

    //// TODO: iterate attributes like this to get all pointdata (and cell data)
    //for a in &piece.data.point
    //{
    //    println!("a = {:?}", a);
    //}

    // Get the data of the first pointdata, assumining it's a scalar

    //let attribute = &piece.data.point[0];
    //let data = match attribute {
    let data: Vec<f32> = match &piece.data.point[0] {
        Attribute::DataArray(DataArray { name, elem, data }) => {
        //Attribute::DataArray(DataArray { elem, data, .. }) => {
            match elem {
                ElementType::Scalars {
                    num_comp,
                    lookup_table,
                    ..
                } => {

                    // TODO: remove
                    println!(
                        //self,
                        "SCALARS {} {} {}",
                        name,
                        ScalarType::from(data.scalar_type()),
                        num_comp
                    );

                    println!(
                        //self,
                        "LOOKUP_TABLE {}",
                        lookup_table.clone().unwrap_or_else(|| String::from("default"))
                    );

                    //println!("data = {:?}", data);

                    // Cast everything to f32.  TODO: other types?
                    match data.scalar_type()
                    {
                        ScalarType::F32 => data.clone().into_vec::<f32>().unwrap(),
                        ScalarType::F64 => data.clone().into_vec::<f64>().unwrap().iter().map(|n| *n as f32).collect(),
                        _ => todo!()
                    }

                }

                // Do vectors, tensors too
                _ => todo!()

            }
        }
        Attribute::Field {..} => todo!()
    };
    
    //println!("data = {:?}", data);

    //****************

    let mut nodes = Vec::with_capacity(points.len());
    for i in 0 .. points.len() / ND
    {
        nodes.push(Node{position: [points[ND*i+0], points[ND*i+1], points[ND*i+2]]});
    }

    let node_buffer = glium::VertexBuffer::new(&display, &nodes).unwrap();

    // This may not handle NaN correctly
    let mut smin = data[0];
    let mut smax = data[0];
    for i in 1 .. data.len()
    {
        if data[i] < smin { smin = data[i]; }
        if data[i] > smax { smax = data[i]; }
    }

    let mut scalar = Vec::with_capacity(data.len());
    for i in 0 .. data.len()
    {
        scalar.push(Scalar{tex_coord: ((data[i] - smin) / (smax - smin)) as f32 });
    }
    let mut sca_buffer = glium::VertexBuffer::new(&display, &scalar).unwrap();

    // Capacity could be set ahead of time for tris with an extra pass over cell types to count
    // triangles
    let mut tris = Vec::new();
    for i in 0 .. piece.cells.types.len()
    {
        if piece.cells.types[i] == CellType::Triangle
        {
            tris.push(vcells.0[ (vcells.1[i as usize] - 3) as usize ] as u32 );
            tris.push(vcells.0[ (vcells.1[i as usize] - 2) as usize ] as u32 );
            tris.push(vcells.0[ (vcells.1[i as usize] - 1) as usize ] as u32 );
        }
    }
    //println!("tris = {:?}", tris);

    // TODO: push other cell types to other buffers.  Draw them with separate calls to
    // target.draw(), share the same vertex and texture buffers

    let tris_buffer = glium::IndexBuffer::new(&display,
        glium::index::PrimitiveType::TrianglesList, &tris).unwrap();

    // TODO: free vtk object and other objects here if possible

    let node_shader_src = r#"
        #version 150

        in vec3 position;
        in float tex_coord;
        out float n_tex_coord;
        out vec3 v_position;
        //flat out vec3 v_normal;

        uniform mat4 matrix;

        void main() {
            n_tex_coord = tex_coord;
            gl_Position = matrix * vec4(position, 1.0);
            v_position = gl_Position.xyz / gl_Position.w;
        }
    "#;

    // TODO: Gouraud or Blinn-Phong shading
    let fragment_shader_src = r#"
        #version 150

        // No lighting
        in float n_tex_coord;
        out vec4 color;
        uniform sampler1D tex;
        void main() {
            color = texture(tex, n_tex_coord);
        }

        //// Blinn-Phong shading
        ////
        //// TODO: duplicate nodes for each cell.  This is apparently the best way to have normals
        //// per cell instead of normals per vertex?  Then pass normals as a new part of the
        //// target.draw() tuple
        ////
        //in float n_tex_coord;
        ////flat in vec3 v_normal;
        //in vec3 v_normal;
        //in vec3 v_position;
        //
        //out vec4 color;
        //
        //uniform vec3 u_light;
        //uniform sampler1D tex;
        //
        //const vec4 ambient_color = vec4(0.2, 0.0, 0.0, 1.0);
        ////const vec3 diffuse_color = vec3(0.6, 0.0, 0.0, 1.0);
        //vec4 diffuse_color = texture(tex, n_tex_coord);
        //const vec4 specular_color = vec4(1.0, 1.0, 1.0, 1.0);
        //
        //void main() {
        //    float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
        //
        //    vec3 camera_dir = normalize(-v_position);
        //    vec3 half_direction = normalize(normalize(u_light) + camera_dir);
        //    float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);
        //
        //    //color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
        //    color = ambient_color + diffuse * diffuse_color + specular * specular_color;
        //    //color = diffuse_color;
        //}

    "#;

    let program = glium::Program::from_source(&display, node_shader_src, fragment_shader_src, None).unwrap();

    let mut t = 0.0;//-0.5;
    event_loop.run(move |event, _, control_flow| {

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        // 60 FPS
        let next_frame_time = std::time::Instant::now() +
            std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        // we update `t`
        //t += 0.01;
        if t > 0.5 {
            t = -0.5;

            // TODO: remove this.  Test only to update sca_buffer w/o updating vertex buffer
            // Can't do "for s in scalar" due to reasons
            //
            // Eventually, add options to cycle through result data arrays

            for s in scalar.iter_mut()
            {
                s.tex_coord = 1.0 - s.tex_coord;
            }
            sca_buffer = glium::VertexBuffer::new(&display, &scalar).unwrap();

        }

        let mut target = display.draw();
        //target.clear_color(0.0, 0.0, 1.0, 1.0);
        target.clear_color(0.322, 0.341, 0.431, 1.0);  // pv

        // TODO: set scale/bounds based on points bounds

        //// scratch/a.vtu
        //let s = 0.05;
        //let tx = -0.8;
        //let ty = -0.8;

        //// rbc
        //let s = 0.2;
        //let tx = -0.0;
        //let ty = -0.0;

        // teapot
        let s = 0.1;
        let tx = -0.0;
        let ty = -0.0;

        //let s = 1.0;
        //let tx = -0.0;
        //let ty = -0.0;

        let uniforms = uniform!
        {
            // TODO: rotations

            // TODO: perspective projection

            // TODO: adjust for window aspect

            matrix: [
                [  s, 0.0, 0.0, 0.0],
                [0.0,   s, 0.0, 0.0],
                [0.0, 0.0,   s, 0.0],
                [ tx,  ty, 0.0, 1.0f32],
            ],

            // Linear sampling works better than the default, especially around texture 0
            tex: glium::uniforms::Sampler::new(&texture)
                    .magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
                    .minify_filter(glium::uniforms::MinifySamplerFilter::Linear),

            model: [
                [0.01, 0.0, 0.0, 0.0],
                [0.0, 0.01, 0.0, 0.0],
                [0.0, 0.0, 0.01, 0.0],
                [0.0, 0.0, 2.0, 1.0f32]
            ],

            view: view_matrix(&[2.0, 1.0, 1.0], &[-2.0, -1.0, 1.0], &[0.0, 1.0, 0.0]),

            perspective: {
                let (width, height) = target.get_dimensions();
                let aspect_ratio = height as f32 / width as f32;

                let fov: f32 = 3.141592 / 3.0;
                let zfar = 1024.0;
                let znear = 0.1;

                let f = 1.0 / (fov / 2.0).tan();

                [
                    [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                    [         0.0         ,     f ,              0.0              ,   0.0],
                    [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                    [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
                ]
            },

            light: [1.4, 0.4, -0.7f32],

        };

        target.draw((&node_buffer, &sca_buffer), &tris_buffer, &program, &uniforms,
                    &Default::default()).unwrap();

        target.finish().unwrap();
    });
}

//==============================================================================

fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [up[1] * f[2] - up[2] * f[1],
             up[2] * f[0] - up[0] * f[2],
             up[0] * f[1] - up[1] * f[0]];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
             f[2] * s_norm[0] - f[0] * s_norm[2],
             f[0] * s_norm[1] - f[1] * s_norm[0]];

    let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
             -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
             -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

    [
        [s_norm[0], u[0], f[0], 0.0],
        [s_norm[1], u[1], f[1], 0.0],
        [s_norm[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}

//==============================================================================

