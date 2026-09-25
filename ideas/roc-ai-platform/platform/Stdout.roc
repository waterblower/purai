import Host

## Stdout operations. I/O failures return IoErr with a diagnostic string.
Stdout := [].{
	write! : Str => Try({}, [IoErr(Str)])
	write! = |value|
		match Host.stdout_write!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	line! : Str => Try({}, [IoErr(Str)])
	line! = |value|
		match Host.stdout_line!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	write_bytes! : List(U8) => Try({}, [IoErr(Str)])
	write_bytes! = |value|
		match Host.stdout_write_bytes!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	flush! : {} => Try({}, [IoErr(Str)])
	flush! = |{}|
		match Host.stdout_flush!({}) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

}
