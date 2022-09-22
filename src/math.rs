
//==============================================================================

use crate::consts::*;

//==============================================================================

// TODO: consider using nalgebra crate for vector/matrix wrapper types with operator overloading

// Can't overload "+" operator because rust makes it impossible by design without a wrapper type :(
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32>
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

pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32>
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

pub fn neg(a: &[f32]) -> Vec<f32>
{
	a.iter().map(|x| -x).collect()
}

pub fn dot(a: &[f32], b: &[f32]) -> f32
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

pub fn norm(a: &[f32]) -> f32
{
	dot(&a, &a).sqrt()
}

pub fn tnorm((w, h): (u32, u32)) -> f32
{
	// Tuple norm
	((w*w) as f32 + (h*h) as f32).sqrt()
}

pub fn normalize(a: &[f32]) -> Vec<f32>
{
	let norm = norm(a);
	a.iter().map(|x| x / norm).collect()
}

pub fn cross(a: &[f32], b: &[f32]) -> [f32; ND]
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

pub fn identity_matrix() -> [[f32; NM]; NM]
{[
	[1.0, 0.0, 0.0, 0.0],
	[0.0, 1.0, 0.0, 0.0],
	[0.0, 0.0, 1.0, 0.0],
	[0.0, 0.0, 0.0, 1.0],
]}

//==============================================================================

pub fn mul_mat4(a: &[[f32; NM]; NM], b: &[[f32; NM]; NM]) -> [[f32; NM]; NM]
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

//==============================================================================

pub fn translate_matrix(m: &[[f32; NM]; NM], u: &[f32]) -> [[f32; NM]; NM]
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

pub fn rotate_matrix(m: &[[f32; NM]; NM], u: &[f32; ND], theta: f32) -> [[f32; NM]; NM]
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

pub fn view_matrix(position: &[f32; ND], direction: &[f32; ND], up: &[f32; ND]) -> [[f32; NM]; NM]
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

pub fn perspective_matrix(fov: f32 , zfar: f32, znear: f32, (width, height): (u32, u32))
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

