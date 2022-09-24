
//****************

use crate::consts::*;
use crate::math::*;
use crate::utils;

//****************

// 3P
use vtkio::{model, model::{Attribute, DataArray, DataSet, ElementType, Vtk}};

//==============================================================================

pub struct Model
{
	// The Model struct contains the geometry and its associated point data
	//
	// For now, this is just a wrapper for the vtkio vtk struct, although it
	// could be generalized for other file formats

	pub points: Vec<f32>,

	// TODO: abstract away from vtkio's types
	pub types  : Vec<model::CellType>,
	pub cells  : Vec<u64>,
	pub offsets: Vec<u64>,

	// Point data arrays
	pub point_data: Vec<Data>,

	// TODO: cell data
}

pub struct Data
{
	// A single point or cell data array
	pub data: Vec<f32>,
	pub name: String,
	pub num_comp: u32,
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
}

//****************

// Split position and texture coordinates into separate arrays.  That way we
// can change texture coordinates (e.g. rescale a colorbar range or load
// a different result) without sending the position arrays to the GPU again

#[derive(Copy, Clone, Debug)]
pub struct Node
{
	// 3D node
	position: [f32; ND]
}
glium::implement_vertex!(Node, position);

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

	pub vertices: glium::VertexBuffer<Node  >,
	pub normals : glium::VertexBuffer<Normal>,
	pub scalar  : glium::VertexBuffer<Scalar>,
	pub indices : glium::index::NoIndices,
}

impl RenderModel
{
	//****************

	pub fn new(m: &Model, facade: &dyn glium::backend::Facade) -> RenderModel
	{
		// Capacity could be set ahead of time for tris with an extra pass over
		// cell types to count triangles
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

		// TODO: push other cell types to other buffers.  Draw them with
		// separate calls to target.draw().  Since vertices are duplicated per
		// cell, there need to be parallel vertex and scalar arrays too.  We
		// could just push every cell type to a big list of tris, but that
		// wouldn't allow correct edge display or advanced filters that treat
		// data at the cell level.

		// TODO: split scalar handling to a separate loop (and eventually
		// a separate fn).  Mesh geometry will only be loaded once, but scalars
		// may be processed multiple times as the user cycles through results to
		// display

		let mut nodes   = Vec::with_capacity(tris.len());
		let mut scalar  = Vec::with_capacity(tris.len());
		let mut normals = Vec::with_capacity(tris.len());

		// Point data index
		let ip = 0;//1;

		// TODO: don't bind scalar like this.  Just call bind_point_data()
		// before returning
		//
		// Get min/max of scalar
		let (smin, smax) = utils::get_bounds(&m.point_data[ip].data);

		for i in 0 .. tris.len() / ND
		{
			// Local array containing the coordinates of the vertices of
			// a single triangle
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

				let s = m.point_data[ip].data[tris[ND*i + j] as usize];
				scalar.push(Scalar{tex_coord:
					((s - smin) / (smax - smin)) as f32 });
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

		RenderModel
		{
			vertices: glium::VertexBuffer::new(facade, &nodes  ).unwrap(),
			normals : glium::VertexBuffer::new(facade, &normals).unwrap(),
			scalar  : glium::VertexBuffer::new(facade, &scalar ).unwrap(),

			indices : glium::index::NoIndices(
				glium::index::PrimitiveType::TrianglesList),
		}
	}

	//****************

	pub fn bind_point_data(&mut self, index: usize, m: &Model,
		facade: &dyn glium::backend::Facade)
	{
		// Select point data array by index to bind for graphical display
		//
		// TODO: add arg "ic" to select vector/tensor component/magnitude.
		// Panic if out of bounds

		let ic = if m.point_data[index].num_comp > 1 {
			1
		} else {
			0
		};

		// TODO: can this be done without looking up tris again, and without
		// saving tris to memory as a struct member?  Can indices be used
		// instead?  If not, refactor this into a fn also used by new().
		//
		// Capacity could be set ahead of time for tris with an extra pass over
		// cell types to count triangles
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

		let mut scalar  = Vec::with_capacity(tris.len());
		let step = m.point_data[index].num_comp as usize;

		// Get min/max of scalar.  TODO: add a magnitude option for vectors (but
		// not tensors).  An extra pass will be needed to calculate
		let (smin, smax) = utils::get_bounds(&(m.point_data[index].data
			.iter().skip(ic).step_by(step).copied().collect::<Vec<f32>>()));
		//let (smin, smax) = utils::get_bounds(&m.point_data[index].data);

		for i in 0 .. tris.len() / ND
		//for i in 0 .. self.vertices.len() / ND
		{
			for j in 0 .. ND
			{
				let s = m.point_data[index].data[
					step * tris[ND*i + j] as usize + ic];

				scalar.push(Scalar{tex_coord:
					((s - smin) / (smax - smin)) as f32 });
			}
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
		// results, etc.
		unimplemented!("multiple pieces");
	}

	let piece = pieces[0].load_piece_data(None).unwrap();

	println!("Number of points = {}", piece.num_points());
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
	m.types   = piece.cells.types;

	//println!("point 0 = {:?}", piece.data.point[0]);
	//println!();

	//let mut name: String = "".to_string();

	let mut point_data = Vec::new();

	// Iterate attributes like this to get all pointdata (TODO: cell data)
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
								num_comp: *num_comp,
							});
					}

					ElementType::Vectors{}
					=>
					{
						println!("Vectors");
						point_data.push(Data
							{
								name: name.to_string(),
								data: data.clone().cast_into::<f32>().unwrap(),
								num_comp: ND as u32,
							});
					}

					ElementType::Tensors{}
					=>
					{
						println!("Tensors");
						point_data.push(Data
							{
								name: name.to_string(),
								data: data.clone().cast_into::<f32>().unwrap(),
								num_comp: ND as u32,
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
								num_comp: *num_comp,
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
		println!("name     = {}", d.name);
		println!("num_comp = {}", d.num_comp);
		println!("len      = {}", d.data.len());
		println!();
	}

	m.point_data = point_data;

	m
}

//==============================================================================

