
//==============================================================================

pub fn get_bounds(x: &[f32]) -> (f32, f32)
{
	// Is this utils or math?
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
	// Rust doesn't do optional args, so we have 2 fn's instead
	fmt_f32(num, 12, 5, 2)
}

pub fn fmt_f32(num: f32, width: usize, precision: usize, exp_width: usize) -> String
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

