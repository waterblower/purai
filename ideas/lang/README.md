# lang

用 Rust 实现的原生语言编译器原型，直接输出 **Apple Silicon / ARM64 macOS 的 Mach-O 可执行文件**。

编译器只有 Rust 标准库依赖，没有第三方 crate。编译 `.lang` 文件时不调用 LLVM、Clang、汇编器、链接器或 `codesign`：ARM64 编码、地址修正、Mach-O 文件生成、SHA-256 和 ad-hoc 签名全部在进程内完成。生成的程序通过 macOS 自带的 dyld / libSystem 加载和输出文字。

## 编译与运行

在本目录执行：

```sh
# 仅此步骤需要 Rust 工具链及其平台构建环境。
cargo build --release --offline

# 后续步骤只需要生成的编译器和 macOS。
./target/release/lang build examples/hello.lang -o build/hello
./build/hello
```

输出：

```text
Hello, world!
```

也支持统一命令：

```sh
./target/release/lang check examples/hello.lang
./target/release/lang run examples/greetings.lang
./target/release/lang --help
```

未指定 `-o` 时，输出到当前工作目录下的 `build/<源文件名，不含扩展名>`。编译结果按临时文件写入后原子替换，避免原地修改已执行文件造成签名缓存问题。编译输出是确定性的。

## 当前语言子集

```rust
fn main() {
    let greeting = "Hello, world!"
    println(greeting)
    other()
}

fn other() {
    print("你好，")
    println("世界！")
}
```

- 无参数、返回 unit 的函数，支持前向调用和嵌套调用。
- UTF-8 字符串字面量、不可变 `let` 字符串绑定、绑定引用及遮蔽。
- `print(str)` / `println(str)`；字符串长度按 UTF-8 字节计数，支持内嵌 NUL。
- 转义：`\n`、`\r`、`\t`、`\0`、`\"`、`\\`。
- `//` 注释；分号可选；标识符目前限 ASCII 字母、数字和下划线。
- 解析、名称解析、调用参数数量检查、字符串绑定检查；诊断包含源文件、行列和指示符。

这是把「源码 → 机器码 → macOS 可执行文件」打通的第一版，**尚未实现**完整的 OCaml 风格类型推导、ADT/GADT、模块系统、数字运算、控制流、动态分配、region/借用检查或通用 FFI。当前所有字符串都是只读常量，`let` 在编译时解析；它们不构成动态内存管理或 region 安全性的验证。

## 实现结构

| 文件 | 职责 |
| --- | --- |
| `src/frontend.rs` | Lexer、parser、名称及基本类型检查，生成平台无关的 checked IR |
| `src/aarch64.rs` | ARM64 指令编码、函数调用修正、常量地址修正、输出 runtime |
| `src/macho.rs` | Mach-O 段/节、dyld 导入绑定、符号表、入口和内置 ad-hoc 签名 |
| `src/sha256.rs` | 为代码签名计算 SHA-256 页哈希 |
| `src/lib.rs` | `check` / `compile` 共享的语义入口与诊断 |
| `src/main.rs` | CLI、文件写入、运行生成的程序 |

生成的映像使用 ASLR/PIE、独立的只读可执行代码段和可写数据段，以及保护空指针地址范围的 `__PAGEZERO`。指针通过 ADRP 相对寻址生成。运行时保留 ARM64 callee-saved 寄存器和 16 字节栈对齐；输出循环处理成功的短写，`write` 返回错误或零进度则以状态码 1 退出。暂不重试 EINTR，也不提供异常展开或调试信息。

输出只支持 `arm64-macos`。文件声明最低 macOS 11，实际验证环境为 macOS 15.4 / arm64；较旧版本尚未实测。ad-hoc 签名用于本机执行，不是 Developer ID 发行签名或公证。原型不读任意 `.o` / `.a`，只链接本语言函数和两个固定的 libSystem 导入。

## 验证

```sh
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
cargo fmt --check
```

ARM64 macOS 上的集成测试会实际执行生成的程序，覆盖嵌套函数调用、Unicode/转义/NUL、跨页长字符串、写入失败、重复构建、错误诊断和清空 `PATH` 后的编译运行。其他平台仅运行平台无关测试。

可选的系统检查工具只用于验证，不参与生成：

```sh
file build/hello
codesign --verify --verbose=4 build/hello
otool -L build/hello
```

## 格式参考

- [Apple Mach-O 定义](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h)
- [Apple CodeDirectory / 签名格式定义](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/cs_blobs.h)
- [Apple ARM64 调用约定](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)
- [SHA-256：FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)

后续可以在 checked IR 中加入类型、region 与资源信息，同时继续复用 CLI、诊断和原生文件输出。
