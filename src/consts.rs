
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

// Augmented matrix size
pub const NM: usize = ND + 1;

// TODO: parameterize # verts per tri

//==============================================================================

