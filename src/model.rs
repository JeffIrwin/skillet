
//****************

use vtkio::model::*;

//==============================================================================

pub struct Model
{
	// For now, this is just a wrapper for the vtkio vtk struct, although it
	// could be generalized for other file formats

	pub points: Vec<f32>,

	// TODO: abstract away from vtkio's types
	pub types  : Vec<vtkio::model::CellType>,
	pub cells  : Vec<u64>,
	pub offsets: Vec<u64>,

	pub pdata: Vec<f32>,

	// TODO: cell data
}

impl Model
{
	pub fn new() -> Model
	{
		Model {

			points: Vec::new(),
			//piece: UnstructuredGridPiece::new(),

			types  : Vec::new(),
			cells  : Vec::new(),
			offsets: Vec::new(),

			pdata: Vec::new(),

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

	//pieces
	//let mut m = Model::new();
	//m

	let piece = pieces[0].load_piece_data(None).unwrap();

	println!("Number of points = {}", piece.num_points());
	println!("Number of cells  = {}", piece.cells.types.len());
	println!();

	//let points = piece.points.cast_into::<f32>().unwrap();
	//m.points = points;

	let mut m: Model = Model::new();
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

