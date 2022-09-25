
//****************

use crate::consts::*;
use crate::math::*;
use crate::utils;

//****************

// 3P
use vtkio::{model::{Attribute, DataArray, DataSet, ElementType, Vtk}};

//==============================================================================

#[derive(PartialEq, Clone, Copy)]
pub enum Type
{
	// There are a few others, but I'm not planning to implement them

	Tri,
	//Quad,
	//Tet,
	Hex,
	//Wedge,
	//Pyramid,

	Invalid, // supported in vtkio but not here
}

pub fn cell_tris(t: Type) -> Vec<usize>
{
	// Return an array of indices of triangle vertices that make up a more
	// complex cell type
	//
	// Ref:  http://www.princeton.edu/~efeibush/viscourse/vtk.pdf
	//
	match t
	{
		Type::Tri => vec!
			[
				0, 1, 2,
			],
		Type::Hex => vec!
			[
				0, 3, 2,   2, 1, 0,
				4, 5, 6,   6, 7, 4,
				0, 1, 5,   5, 4, 0,
				1, 2, 6,   6, 5, 1,
				2, 3, 7,   7, 6, 2,
				0, 4, 7,   7, 3, 0,
			],

		Type::Invalid => vec![],
	}
}

//==============================================================================

pub fn cell_num_verts(t: Type) -> usize
{
	match t
	{
		Type::Tri => 3,
		Type::Hex => 8,

		Type::Invalid => 0,
	}
}

pub fn cell_edges(t: Type) -> Vec<usize>
{
	// Return an array of edge vertices that make up a more complex cell type
	//
	// TODO: implement display
	match t
	{
		Type::Tri => vec!
			[
				0, 1,
				1, 2,
				2, 0,
			],
		Type::Hex => vec!
			[
				0, 1,
				1, 2,
				2, 3,
				3, 0,
				4, 5,
				5, 6,
				6, 7,
				7, 4,
				0, 4,
				1, 5,
				2, 6,
				3, 7,
			],

		Type::Invalid => vec![],
	}
}

//==============================================================================

pub struct Model
{
	// The Model struct contains the geometry and its associated point data
	//
	// For now, this is just a wrapper for the vtkio vtk struct, although it
	// could be generalized for other file formats

	pub points: Vec<f32>,

	pub types  : Vec<Type>,
	pub cells  : Vec<u64>,
	pub offsets: Vec<u64>,

	// Point data arrays
	pub point_data: Vec<Data>,

	// TODO: cell data.  "./res/hex.vtu" has cell IDs as cell data
}

pub struct Data
{
	// A single point or cell data array
	pub data: Vec<f32>,
	pub name: String,
	pub num_comp: usize,
}

impl Model
{
	pub fn new() -> Model
	{
		Model
		{
			points: Vec::new(),
			//piece: UnstructuredGridPiece::new(),

			types  : Vec::new(),
			cells  : Vec::new(),
			offsets: Vec::new(),

			point_data: Vec::new(),
		}
	}

	//****************

	pub fn tris(&self) -> Vec<u64>
	{
		// TODO: can this be done without looking up tris again, and without
		// saving tris to memory as a struct member?  Can RenderModel indices be
		// used instead?  Tri lookup is only performed on loading and on
		// bind_point_data() for changing data arrays, so it doesn't seem worth
		// saving in memory.

		// Capacity could be set ahead of time for tris with an extra pass over
		// cell types to count triangles
		let mut tris = Vec::new();
		for i in 0 .. self.types.len() as usize
		{
			let t = cell_tris(self.types[i]);
			let nv = cell_num_verts(self.types[i]);
			let nt = t.len() / NT;

			//println!("nt = {}", nt);

			for it in 0 .. nt as usize
			{
				let i0 = self.offsets[i] as usize - nv + t[NT * it + 0];
				let i1 = self.offsets[i] as usize - nv + t[NT * it + 1];
				let i2 = self.offsets[i] as usize - nv + t[NT * it + 2];

				tris.push(self.cells[i0]);
				tris.push(self.cells[i1]);
				tris.push(self.cells[i2]);
			}
		}
		tris
	}
}

//****************

// Split position and texture coordinates into separate arrays.  That way we can
// change texture coordinates (e.g. rescale a colorbar range or load a different
// data array) without sending the position arrays to the GPU again

#[derive(Copy, Clone, Debug)]
pub struct Vert
{
	// 3D vert
	position: [f32; ND]
}
glium::implement_vertex!(Vert, position);

// Even vectors and tensors are be rendered as "scalars", since you can
// only colormap one component (or magnitude) at a time, which is a scalar
#[derive(Copy, Clone)]
pub struct Scalar
{
	tex_coord: f32,
}
glium::implement_vertex!(Scalar, tex_coord);

#[derive(Copy, Clone, Debug)]
pub struct Normal
{
	normal: [f32; ND]
}
glium::implement_vertex!(Normal, normal);

//==============================================================================

pub struct RenderModel
{
	// The RenderModel struct is an interface layer between the Model and
	// glium's GL array/buffer object bindings

	pub vertices: glium::VertexBuffer<Vert  >,
	pub normals : glium::VertexBuffer<Normal>,
	pub scalar  : glium::VertexBuffer<Scalar>,
	pub indices : glium::index::NoIndices,
}

impl RenderModel
{
	//****************

	pub fn new(m: &Model, facade: &dyn glium::backend::Facade) -> RenderModel
	{
		let tris = m.tris();

		// Split scalar handling to a separate fn.  Mesh geometry will only be
		// loaded once, but scalars are processed multiple times as the user
		// cycles through data to display

		// You would think that normals could be 1/3 this size, but they need to
		// be duplicated for each vertex of a triangle for sharp edge shading

		let mut verts   = Vec::with_capacity(tris.len());
		let mut normals = Vec::with_capacity(tris.len());
		let scalar  = Vec::new();

		for i in 0 .. tris.len() / ND
		{
			// Local array containing the coordinates of the vertices of
			// a single triangle
			let mut p: [f32; ND*ND] = [0.0; ND*ND];

			// Some of these ND's should be NT's, not that it makes a difference
			for j in 0 .. ND
			{
				p[ND*j + 0] = m.points[ND*tris[ND*i + j] as usize + 0];
				p[ND*j + 1] = m.points[ND*tris[ND*i + j] as usize + 1];
				p[ND*j + 2] = m.points[ND*tris[ND*i + j] as usize + 2];

				verts.push(Vert{position:
					[
						p[ND*j + 0],
						p[ND*j + 1],
						p[ND*j + 2],
					]});
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

		//println!("vert   0 = {:?}", verts[0]);
		//println!("vert   1 = {:?}", verts[1]);
		//println!("vert   2 = {:?}", verts[2]);
		//println!("normal 0 = {:?}", normals[0]);

		let mut render_model = RenderModel
		{
			vertices: glium::VertexBuffer::new(facade, &verts  ).unwrap(),
			normals : glium::VertexBuffer::new(facade, &normals).unwrap(),
			scalar  : glium::VertexBuffer::new(facade, &scalar ).unwrap(),

			indices : glium::index::NoIndices(
				glium::index::PrimitiveType::TrianglesList),
		};

		render_model.bind_point_data(0, 0, &m, facade);

		render_model
	}

	//****************

	pub fn bind_point_data(&mut self, index: usize, comp: usize, m: &Model,
		facade: &dyn glium::backend::Facade)
	{
		// Select point data array by index to bind for graphical display

		if comp >= m.point_data[index].num_comp
		{
			panic!("Component is out of bounds");
		}

		let tris = m.tris();
		let mut scalar  = Vec::with_capacity(tris.len());
		let step = m.point_data[index].num_comp;

		// Get min/max of scalar.  TODO: add a magnitude option for vectors (but
		// not tensors).  An extra pass will be needed to calculate
		let (smin, smax) = utils::get_bounds(&(m.point_data[index].data
			.iter().skip(comp).step_by(step).copied().collect::<Vec<f32>>()));
		//let (smin, smax) = utils::get_bounds(&m.point_data[index].data);

		for i in 0 .. tris.len()
		{
			let s = m.point_data[index].data[step * tris[i] as usize + comp];

			scalar.push(Scalar{tex_coord:
				((s - smin) / (smax - smin)) as f32 });
		}

		self.scalar = glium::VertexBuffer::new(facade, &scalar).unwrap();
	}

	//****************
}

//==============================================================================

pub fn import(f: std::path::PathBuf)
	//-> Vec<Piece<UnstructuredGridPiece>>
	-> Model
{
	println!("Importing VTK file \"{}\"", f.display());
	println!();

	let vtk = Vtk::import(&f).expect(&format!(
			"Failed to load file: {:?}", f));

	//let file_out = PathBuf::from("./scratch/ascii.vtu");
	//vtk.export_ascii(&file_out)
	//	.expect(&format!("Failed to save file: {:?}", file_out));
	//return;

	// TODO: match UnstructuredGrid vs PolyData, etc.
	//
	// VTK polydata files (or other piece types) can be saved as
	// UnstructuredGrid (.vtu) in ParaView with Filters -> Alphabetical ->
	// Append datasets, in the mean time until I implement polydata natively
	// here
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
		// To do?  Render each piece as if it's a totally separate VTK file.
		// They could have unrelated numbers of points, number and type of
		// data arrays, etc.
		unimplemented!("multiple pieces");
	}

	let piece = pieces[0].load_piece_data(None).unwrap();
	let num_points = piece.num_points();

	println!("Number of points = {}", num_points);
	println!("Number of cells  = {}", piece.cells.types.len());
	println!();

	//let points = piece.points.cast_into::<f32>().unwrap();
	//m.points = points;

	// TODO: instead of making a new empty Model here and setting members one at
	// a time, set them all at the end (like RenderModel) and eliminate the
	// new() fn
	let mut m = Model::new();

	m.points = piece.points.cast_into::<f32>().unwrap();

	//println!("m.points = {:?}", m.points);
	//println!();

	// Convert legacy into XML so we don't have to match conditionally
	let cells = piece.cells.cell_verts.into_xml();

	//println!("connectivity = {:?}", cells.0);
	//println!("types        = {:?}", piece.cells.types);
	//println!("offsets      = {:?}", cells.1);
	//println!();

	// In vtkio, cells.0 is the actual connectivity, and cells.1 is the offset
	m.cells   = cells.0;
	m.offsets = cells.1;

	//m.types   = piece.cells.types;

	// Abstract away from vtkio's type enum
	m.types = Vec::with_capacity(piece.cells.types.len());
	for t in piece.cells.types
	{
		m.types.push(match t
		{
			vtkio::model::CellType::Triangle   => Type::Tri,
			vtkio::model::CellType::Hexahedron => Type::Hex,

			//vtkio::model::CellType::Line       => Type::Unsupported,
			//vtkio::model::CellType::PolyVertex => Type::Unsupported,
			//vtkio::model::CellType::Vertex     => Type::Unsupported,

			_ => Type::Invalid,
		});
	}

	//println!("point 0 = {:?}", piece.data.point[0]);
	//println!();

	//let mut name: String = "".to_string();

	let mut point_data = Vec::new();

	// Iterate attributes like this to get all pointdata (TODO: make this a fn
	// and parse cell data too)
	for attrib in &piece.data.point
	{
		println!("Attribute:");
		//println!("attrib = {:?}", attrib);

		// Get the contents of the pointdata array.  This is based on
		// write_attrib() from vtkio/src/writer.rs

		match attrib
		{
			Attribute::DataArray(DataArray {elem, data, name}) =>
			{
				let data_len = data.len();
				match elem
				{
					ElementType::Scalars{num_comp, ..}
					=>
					{
						println!("Scalars");

						// Cast everything to f32

						point_data.push(Data
							{
								name: name.to_string(),
								data: data.clone().cast_into::<f32>().unwrap(),
								num_comp: *num_comp as usize,
							});
					}

					ElementType::Vectors{}
					=>
					{
						// Vector num_comp should always be 3, but calculate it
						// anyway
						println!("Vectors");
						point_data.push(Data
							{
								name: name.to_string(),
								data: data.clone().cast_into::<f32>().unwrap(),
								num_comp: data_len / num_points,
							});
					}

					ElementType::Tensors{}
					=>
					{
						// Tensor num_comp may be 6 or 9
						println!("Tensors");
						point_data.push(Data
							{
								name: name.to_string(),
								data: data.clone().cast_into::<f32>().unwrap(),
								num_comp: data_len / num_points,
							});
					}

					ElementType::Generic(num_comp)
					=>
					{
						println!("Generic");
						point_data.push(Data
							{
								name: name.to_string(),
								data: data.clone().cast_into::<f32>().unwrap(),
								num_comp: *num_comp as usize,
							});
					}

					//ElementType::ColorScalars{..}
					//=>
					//{
					//	println!("ColorScalars");
					//}

					//ElementType::LookupTable{..}
					//=>
					//{
					//	println!("LookupTable");
					//}

					//ElementType::Normals{..}
					//=>
					//{
					//	println!("Normals");
					//}

					//ElementType::TCoords{..}
					//=>
					//{
					//	println!("TCoords");
					//}

					// Just ignore and don't push anything that I haven't
					// handled
					_ => ()
				}
			}
			Attribute::Field {..}
					=> unimplemented!("field attribute for point data")
		};
		println!();
	}

	// TODO: display name in legend.  Apparently I need another crate for GL
	// text display

	println!("point_data.len() = {}", point_data.len());
	for d in &point_data
	{
		println!("\tname     = {}", d.name);
		println!("\tnum_comp = {}", d.num_comp);
		println!("\tlen      = {}", d.data.len());
		println!();
	}

	m.point_data = point_data;

	m
}

//==============================================================================

