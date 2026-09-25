import Host

## Stdin operations. I/O failures return IoErr with a diagnostic string.
Stdin := [].{
	line! : {} => Try({ eof : Bool, text : Str }, [IoErr(Str)])
	line! = |{}|
		match Host.stdin_line!({}) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	read_text! : {} => Try(Str, [IoErr(Str)])
	read_text! = |{}|
		match Host.stdin_read_text!({}) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

	read_bytes! : {} => Try(List(U8), [IoErr(Str)])
	read_bytes! = |{}|
		match Host.stdin_read_bytes!({}) {
			Ok(result) => Ok(result)
			Err(message) => Err(IoErr(message))
		}

}
