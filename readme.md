This is a young project. Me being a single person attacking the whole AI inference problem is a huge undertaking.

Therefore, I am starting from the easier problem, quantization of gguf models, which forces me to learn the gguf formats inside-out.

The first usage software produced by the project is
```
ai quantize -m model-path.gguf -o output-path.gguf
```
and
```
ai print -m model-path.gguf
```

To install the cli, currently you can only build from source
```
zig build ai
```

### Credit
This project is inspired by [llama2.c](https://github.com/karpathy/llama2.c) from [Andrej Karpathy](https://github.com/karpathy) who probably taught me most of what I know about AI to this point.

Huge thanks to [Andrew Kelly](https://andrewkelley.me/) who created [Zig Programming Language](https://ziglang.org/) which allowed me to create this project.

P.S. Is Andre[x] the name for excellent engineer now?
