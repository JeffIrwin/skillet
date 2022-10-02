
const EDGE: &str = include_str!("./edge.glsl");
const FRAG: &str = include_str!("./frag.glsl");
const VERT: &str = include_str!("./vert.glsl");

// TODO: add Gouraud option in addition to current Blinn-Phong frag shader

pub fn edge(facade: &dyn glium::backend::Facade) -> glium::Program
{
	glium::Program::from_source(facade, VERT, EDGE, None).unwrap()
}

pub fn face(facade: &dyn glium::backend::Facade) -> glium::Program
{
	glium::Program::from_source(facade, VERT, FRAG, None).unwrap()
}

