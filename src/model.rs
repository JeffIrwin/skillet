
//****************

use crate::consts::*;
use crate::math::*;

//****************

// 3P
use vtkio::{model, model::{Attribute, DataArray, DataSet, ElementType, Vtk}};

//==============================================================================

pub struct Model
{
	// For now, this is just a wrapper for the vtkio vtk struct, although it
	// could be generalized for other file formats

	pub points: Vec<f32>,

	// TODO: abstract away from vtkio's types
	pub types  : Vec<model::CellType>,
	pub cells  : Vec<u64>,
	pub offsets: Vec<u64>,

	pub pdata: Vec<f32>,

	// TODO: cell data
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

			pdata: Vec::new(),
		}
	}
}

//****************

//#[macro_use]
//extern crate glium;

#[derive(Copy, Clone, Debug)]
pub struct Node
{
	// 3D node
	position: [f32; ND]
}
glium::implement_vertex!(Node, position);

#[derive(Copy, Clone, Debug)]
pub struct Node2
{
	// 2D node for background
	position2: [f32; N2],
	tex_coord: f32,
}
glium::implement_vertex!(Node2, position2, tex_coord);

// Even vectors and tensors will be rendered as "scalars", since you can
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
	pub vertices: glium::VertexBuffer<Node>,
}

impl RenderModel
{
	pub fn new(display: &dyn glium::backend::Facade, m: &Model) -> RenderModel
	//pub fn new(display: glium::backend::glutin::Display, m: Model) -> RenderModel
	{

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

				//let s = m.pdata[tris[ND*i + j] as usize];
				//scalar.push(Scalar{tex_coord: ((s-smin) / (smax-smin)) as f32 });
			}

			let p01 = sub(&p[3..6], &p[0..3]);
			let p02 = sub(&p[6..9], &p[0..3]);

			let nrm = normalize(&cross(&p01, &p02));

			// Use inward normal for RH coordinate system
			for _j in 0 .. ND
			{
				//normals.push(Normal{normal:
				//	[
				//		-nrm[0],
				//		-nrm[1],
				//		-nrm[2],
				//	]
				//});
			}
		}

		RenderModel
		{
			//vertices: glium::VertexBuffer::empty().unwrap(),

			//vertices: glium::VertexBuffer::new(&display, &nodes).unwrap(),
			vertices: glium::VertexBuffer::new(display, &nodes).unwrap(),
		}
	}
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

	//// TODO: iterate attributes like this to get all pointdata (and cell data)
	//for a in &piece.data.point
	//{
	//	println!("a = {:?}", a);
	//}

	// Get the contents of the first pointdata array, assumining it's a scalar.
	// This is based on write_attrib() from vtkio/src/writer.rs
	let (name, pdata) = match &piece.data.point[0]
	{
		Attribute::DataArray(DataArray {elem, data, name}) =>
		{
			match elem
			{
				ElementType::Scalars{..}
				=>
				{
					// Cast everything to f32
					(name, data.clone().cast_into::<f32>().unwrap())
				}

				// TODO: vectors, tensors
				_ => todo!()
			}
		}
		Attribute::Field {..}
				=> unimplemented!("field attribute for point data")
	};

	// TODO: display in legend
	println!("Point data name = {}", name);

	//println!("pdata = {:?}", pdata);

	m.pdata = pdata;

	m
}

//==============================================================================

