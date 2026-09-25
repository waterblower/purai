import Host

## Stderr operations. I/O failures return IoErr with a diagnostic string.
Stderr := [].{
	write! : Str => Try({}, [IoErr(Str)])
	write! = |value|
		match Host.stderr_write!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	line! : Str => Try({}, [IoErr(Str)])
	line! = |value|
		match Host.stderr_line!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	write_bytes! : List(U8) => Try({}, [IoErr(Str)])
	write_bytes! = |value|
		match Host.stderr_write_bytes!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	flush! : {} => Try({}, [IoErr(Str)])
	flush! = |{}|
		match Host.stderr_flush!({}) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

}
