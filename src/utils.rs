
use std::env;

//==============================================================================

pub fn get_exe_base(default: &str) -> String
{
	// Safely get the basename of this exe or return a default

	let exe = match env::current_exe().ok()
	{
		Some(inner) => inner,
		None        => return default.to_string(),
	};
	let exe_opt = match exe.file_name()
	{
		Some(inner) => inner,
		None        => return default.to_string(),
	};
	match exe_opt.to_str()
	{
		Some(inner) => inner.to_string(),
		None        => default.to_string(),
	}
}

//==============================================================================

pub fn get_bounds(x: &[f32]) -> (f32, f32)
{
	// Does this belong in utils or math?
	//
	// This may not handle NaN correctly

	let mut xmin = x[0];
	let mut xmax = x[0];
	for i in 1 .. x.len()
	{
		if x[i] < xmin { xmin = x[i]; }
		if x[i] > xmax { xmax = x[i]; }
	}
	(xmin, xmax)
}

//==============================================================================

pub fn ff32(num: f32) -> String
{
	// Format float in scientific notation for tabular output
	//
	// Rust doesn't do optional args, so we have 2 fn's instead
	fmt_f32(num, 12, 5, 2)
}

pub fn fmt_f32(num: f32, width: usize, precision: usize, exp_width: usize)
		-> String
{
	// https://stackoverflow.com/questions/65264069/alignment-of-floating-point-numbers-printed-in-scientific-notation

	let mut num = format!("{:.precision$e}", num, precision = precision);

	// Safe to `unwrap` as `num` is guaranteed to contain `'e'`
	let exp = num.split_off(num.find('e').unwrap());

	let (sign, exp) = if exp.starts_with("e-")
	{
		('-', &exp[2..])
	}
	else
	{
		('+', &exp[1..])
	};
	num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_width));

	format!("{:>width$}", num, width = width)
}

//==============================================================================

