use std::fs::OpenOptions;
use std::io::{self, BufRead, Read, Write};

/// Strip exactly one LF or CRLF terminator, preserving all other whitespace.
/// EOF is separate from an empty line, including after an unterminated final line.
pub fn read_line(reader: &mut impl BufRead) -> io::Result<(String, bool)> {
    let mut text = String::new();
    let eof = reader.read_line(&mut text)? == 0;
    if text.ends_with('\n') {
        text.pop();
        if text.ends_with('\r') {
            text.pop();
        }
    }
    Ok((text, eof))
}

pub fn read_bytes(reader: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub fn read_text(reader: &mut impl Read) -> io::Result<String> {
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    Ok(text)
}

pub fn write(writer: &mut impl Write, bytes: &[u8], newline: bool) -> io::Result<()> {
    writer.write_all(bytes)?;
    if newline {
        writer.write_all(b"\n")?;
    }
    Ok(())
}

pub fn write_file(path: &str, bytes: &[u8], append: bool) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(path)?;
    file.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn blank_lines_eof_and_terminators_are_distinct() {
        let mut input = Cursor::new(b"\nhello\r\nkeep\r\r\nlast\r");
        for expected in ["", "hello", "keep\r", "last\r"] {
            assert_eq!(read_line(&mut input).unwrap(), (expected.into(), false));
        }
        assert_eq!(read_line(&mut input).unwrap(), (String::new(), true));
        assert_eq!(read_line(&mut input).unwrap(), (String::new(), true));
    }

    #[test]
    fn invalid_utf8_is_an_error_only_for_text() {
        assert_eq!(
            read_text(&mut Cursor::new([255])).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            read_line(&mut Cursor::new([255, 10])).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(read_bytes(&mut Cursor::new([0, 255])).unwrap(), [0, 255]);
    }

    #[test]
    fn short_writes_are_completed_and_failures_propagate() {
        struct ShortWriter(Vec<u8>);
        impl Write for ShortWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                let n = bytes.len().min(2);
                self.0.extend_from_slice(&bytes[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut short = ShortWriter(Vec::new());
        write(&mut short, b"abcde", true).unwrap();
        assert_eq!(short.0, b"abcde\n");
        let mut full = [0; 1];
        assert_eq!(
            write(&mut full.as_mut_slice(), b"abc", false)
                .unwrap_err()
                .kind(),
            io::ErrorKind::WriteZero
        );
    }
}
