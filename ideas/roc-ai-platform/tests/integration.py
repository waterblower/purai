#!/usr/bin/env python3
"""Compile real Roc applications and assert their observable process behavior."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "integration"


class PlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([str(ROOT / "build.sh")], cwd=ROOT, check=True)
        BIN.mkdir(parents=True, exist_ok=True)
        sources = list((ROOT / "examples").glob("*.roc")) + [ROOT / "tests/io.roc"]
        for source in sources:
            subprocess.run(
                [str(ROOT / "scripts/roc.sh"), "build", str(source),
                 "--output=" + str(BIN / source.stem), "--no-cache"],
                cwd=ROOT, check=True, timeout=120,
            )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="roc-io-", dir=BIN)
        self.cwd = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def run_app(self, app, *args, data=b"", code=0):
        result = subprocess.run([str(BIN / app), *map(str, args)],
                                input=data, capture_output=True, cwd=self.cwd, timeout=15)
        self.assertEqual(result.returncode, code, result.stderr.decode(errors="replace"))
        return result

    def test_hello_and_arguments(self):
        self.assertEqual(self.run_app("hello").stdout, b"Hello from Roc!\n")
        self.assertIn(b"Usage:", self.run_app("cat", code=2).stderr)
        self.run_app("io", "exit", code=7)

    def test_stream_writes_and_flushes(self):
        result = self.run_app("io", "streams")
        self.assertEqual(result.stdout, b"output\n\x00\xff\n")
        self.assertEqual(result.stderr, b"error\n\x00\xfe\n")

    def test_lines_preserve_blanks_whitespace_and_final_line(self):
        data = "\nhello\r\n\n日本語  \nkeep\r\r\nfinal".encode()
        expected = "\nhello\n\n日本語  \nkeep\r\nfinal\n".encode()
        self.assertEqual(self.run_app("echo", data=data).stdout, expected)
        self.assertEqual(self.run_app("echo").stdout, b"")
        self.assertEqual(self.run_app("echo", data=b"\n\n").stdout, b"\n\n")

    def test_binary_and_text_stdin(self):
        data = bytes(range(256)) * 1000
        self.assertEqual(self.run_app("io", "bytes", data=data).stdout, data)
        text = ("long input 日本語\n" * 1000).encode()
        self.assertEqual(self.run_app("io", "text", data=text).stdout, text)
        self.assertEqual(self.run_app("io", "text").stdout, b"")
        self.assertEqual(self.run_app("io", "bytes").stdout, b"")

    def test_mixing_line_and_byte_reads_keeps_buffered_input(self):
        self.assertEqual(self.run_app("io", "mixed-input", data=b"line\n\x00\xfftail").stdout,
                         b"line\n\x00\xfftail")

    def test_invalid_utf8_is_reported(self):
        for app, args in [("echo", []), ("io", ["text"])]:
            result = self.run_app(app, *args, data=b"\xff\n", code=1)
            self.assertIn(b"IoErr", result.stderr)
            self.assertIn(b"InvalidData", result.stderr)

    def test_text_files_append_truncate_unicode_and_empty(self):
        path = self.cwd / "space 日本語.txt"
        content = "text 日本語\x00\n" * 1000
        # argv cannot carry NUL; stdin and file contents can.
        self.run_app("save_input", path, data=content.encode())
        self.assertEqual(self.run_app("cat", path).stdout, content.encode())
        self.run_app("io", "append", path, "tail")
        self.assertEqual(path.read_bytes(), content.encode() + b"tail")
        self.run_app("io", "write", path, "short")
        self.assertEqual(path.read_bytes(), b"short")
        self.run_app("io", "write", path, "")
        self.assertEqual(self.run_app("cat", path).stdout, b"")
        self.run_app("io", "append", "new.txt", "created")
        self.assertEqual((self.cwd / "new.txt").read_bytes(), b"created")

    def test_binary_files_and_append(self):
        source, dest = self.cwd / "source.bin", self.cwd / "destination.bin"
        data = bytes(range(256)) * 1000
        source.write_bytes(data)
        self.run_app("copy", source, dest)
        self.assertEqual(dest.read_bytes(), data)
        self.run_app("io", "append-bytes", dest)
        self.assertEqual(dest.read_bytes(), data + b"\x00\xff\n")
        self.run_app("io", "append-bytes", "new.bin")
        self.assertEqual((self.cwd / "new.bin").read_bytes(), b"\x00\xff\n")
        source.write_bytes(b"")
        self.run_app("copy", source, dest)
        self.assertEqual(dest.read_bytes(), b"")

    def test_file_errors_are_recoverable(self):
        result = self.run_app("cat", "missing.txt", code=1)
        self.assertIn(b"NotFound", result.stderr)
        self.assertEqual(self.run_app("io", "recover", "missing.txt").stdout, b"recovered\n")
        (self.cwd / "invalid.txt").write_bytes(b"\xff")
        self.assertIn(b"InvalidData", self.run_app("cat", "invalid.txt", code=1).stderr)
        self.assertIn(b"IoErr", self.run_app("io", "write", "absent/child", "value", code=1).stderr)
        self.run_app("cat", self.cwd, code=1)
        self.run_app("io", "remove", "missing.txt", code=1)

    def test_directories_exists_and_remove(self):
        self.assertEqual(self.run_app("io", "exists", "missing").stdout, b"no\n")
        self.run_app("io", "mkdir", "a/b/c")
        self.run_app("io", "mkdir", "a/b/c")
        self.assertTrue((self.cwd / "a/b/c").is_dir())
        self.assertEqual(self.run_app("io", "exists", "a").stdout, b"yes\n")
        self.run_app("io", "write", "a/b/c/data", "test")
        self.assertEqual(self.run_app("io", "exists", "a/b/c/data").stdout, b"yes\n")
        self.run_app("io", "remove", "a/b/c/data")
        self.assertFalse((self.cwd / "a/b/c/data").exists())
        self.run_app("io", "remove", "a", code=1)

    def test_repeated_host_calls_and_shared_roc_values(self):
        data = bytes(range(256)) * 100
        path = self.cwd / "long-allocated-path-for-reference-counting.bin"
        path.write_bytes(data)
        self.run_app("io", "repeat", path)
        self.assertEqual((self.cwd / "copy-one.bin").read_bytes(), data)
        self.assertEqual((self.cwd / "copy-two.bin").read_bytes(), data)

    @unittest.skipUnless(os.name == "posix", "requires POSIX pipe semantics")
    def test_broken_stdout_is_reported_without_success_exit(self):
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        try:
            result = subprocess.run([str(BIN / "hello")], stdout=write_fd,
                                    stderr=subprocess.PIPE, cwd=self.cwd, timeout=15)
        finally:
            os.close(write_fd)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"BrokenPipe", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
