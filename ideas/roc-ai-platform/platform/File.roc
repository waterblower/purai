import Host

## File operations. I/O failures return IoErr with a diagnostic string.
File := [].{
	read_text! : Str => Try(Str, [IoErr(Str)])
	read_text! = |value|
		match Host.file_read_text!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	read_bytes! : Str => Try(List(U8), [IoErr(Str)])
	read_bytes! = |value|
		match Host.file_read_bytes!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	write_text! : Str, Str => Try({}, [IoErr(Str)])
	write_text! = |path, value|
		match Host.file_write_text!(path, value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	append_text! : Str, Str => Try({}, [IoErr(Str)])
	append_text! = |path, value|
		match Host.file_append_text!(path, value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	write_bytes! : Str, List(U8) => Try({}, [IoErr(Str)])
	write_bytes! = |path, value|
		match Host.file_write_bytes!(path, value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	append_bytes! : Str, List(U8) => Try({}, [IoErr(Str)])
	append_bytes! = |path, value|
		match Host.file_append_bytes!(path, value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	exists! : Str => Try(Bool, [IoErr(Str)])
	exists! = |value|
		match Host.file_exists!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	remove! : Str => Try({}, [IoErr(Str)])
	remove! = |value|
		match Host.file_remove!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	create_dir_all! : Str => Try({}, [IoErr(Str)])
	create_dir_all! = |value|
		match Host.file_create_dir_all!(value) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

}
