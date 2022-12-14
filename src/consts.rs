
//==============================================================================

// Global constants

pub const ME: &str = "Skillet";
pub const MAJOR: &str = env!("CARGO_PKG_VERSION_MAJOR");
pub const MINOR: &str = env!("CARGO_PKG_VERSION_MINOR");
pub const PATCH: &str = env!("CARGO_PKG_VERSION_PATCH");

// I can't figure out how to cat const strs to another const str, but I can make
// a macro
#[macro_export]
macro_rules! mev
{
	() =>
	{
		format!("{} {}.{}.{}", ME, MAJOR, MINOR, PATCH)
	};
}

pub use std::f32::consts::PI;

// Number of dimensions
pub const ND: usize = 3;
pub const N2: usize = 2;

// Number of vertices per triangle
pub const NT: usize = 3;

// Number of vertices per edge (line)
pub const NE: usize = 2;

// Augmented matrix size
pub const NM: usize = ND + 1;

//==============================================================================

