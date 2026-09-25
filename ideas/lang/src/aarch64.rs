//! Small ARM64 encoder and internal relocation pass. Offsets are relative to
//! the Mach-O image, so ADRP/ADD and BL continue working with ASLR enabled.
use crate::frontend::{Op, Program};

pub const CODE_OFFSET: usize = 4096;

enum Target {
    Function(usize),
    Print,
    Local(usize),
}
enum Fixup {
    Branch {
        at: usize,
        opcode: u32,
        bits: u32,
        shift: u32,
        target: Target,
    },
    String {
        at: usize,
        index: usize,
    },
}

pub struct Code {
    pub bytes: Vec<u8>,
    pub instruction_bytes: usize,
    pub constants_bytes: usize,
    imports: Vec<(usize, usize)>,
}

impl Code {
    pub fn bind_imports(&mut self, got_offset: usize) -> Result<(), String> {
        for &(at, slot) in &self.imports {
            let pc = CODE_OFFSET + at * 4;
            let address = got_offset + slot * 8;
            let adrp = adrp(16, pc, address)?;
            let ldr = 0xf9400000 | (((address & 4095) as u32 / 8) << 10) | (16 << 5) | 16;
            self.bytes[at * 4..at * 4 + 4].copy_from_slice(&adrp.to_le_bytes());
            self.bytes[at * 4 + 4..at * 4 + 8].copy_from_slice(&ldr.to_le_bytes());
        }
        Ok(())
    }
}

fn adrp(register: u32, pc: usize, address: usize) -> Result<u32, String> {
    let pages = (address >> 12) as i64 - (pc >> 12) as i64;
    if !(-(1 << 20)..(1 << 20)).contains(&pages) {
        return Err("program is too large for an ARM64 page-relative address".into());
    }
    let imm = (pages as u32) & 0x1f_ffff;
    Ok(0x90000000 | ((imm & 3) << 29) | ((imm >> 2) << 5) | register)
}

#[derive(Default)]
struct Encoder {
    words: Vec<u32>,
    strings: Vec<Vec<u8>>,
    fixups: Vec<Fixup>,
    imports: Vec<(usize, usize)>,
    locals: Vec<usize>,
}

impl Encoder {
    fn emit(&mut self, word: u32) {
        self.words.push(word);
    }
    fn position(&self) -> usize {
        self.words.len()
    }
    fn prologue(&mut self) {
        self.emit(0xa9bf7bfd); // stp x29, x30, [sp, #-16]!
        self.emit(0x910003fd); // mov x29, sp
    }
    fn epilogue(&mut self) {
        self.emit(0xa8c17bfd); // ldp x29, x30, [sp], #16
        self.emit(0xd65f03c0); // ret
    }
    fn immediate(&mut self, register: u32, value: u64) {
        self.emit(0xd2800000 | (((value & 0xffff) as u32) << 5) | register);
        for part in 1..4 {
            let bits = ((value >> (part * 16)) & 0xffff) as u32;
            if bits != 0 {
                self.emit(0xf2800000 | (part << 21) | (bits << 5) | register);
            }
        }
    }
    fn branch(&mut self, opcode: u32, bits: u32, shift: u32, target: Target) {
        self.fixups.push(Fixup::Branch {
            at: self.position(),
            opcode,
            bits,
            shift,
            target,
        });
        self.emit(0);
    }
    fn call(&mut self, target: Target) {
        self.branch(0x94000000, 26, 0, target);
    }
    fn import(&mut self, slot: usize) {
        self.imports.push((self.position(), slot));
        self.emit(0); // adrp x16, import page
        self.emit(0); // ldr x16, [x16, page offset]
        self.emit(0xd63f0200); // blr x16
    }
    fn new_label(&mut self) -> usize {
        self.locals.push(usize::MAX);
        self.locals.len() - 1
    }
    fn label(&mut self, id: usize) {
        self.locals[id] = self.position();
    }
    fn string(&mut self, value: &[u8]) {
        let index = self.strings.len();
        self.strings.push(value.to_vec());
        self.fixups.push(Fixup::String {
            at: self.position(),
            index,
        });
        self.emit(0); // adrp x0, literal page
        self.emit(0); // add x0, x0, page offset
        self.immediate(1, value.len() as u64);
        self.call(Target::Print);
    }

    // Internal ABI: x0 = bytes, x1 = length. Preserve Darwin callee-saved
    // registers and 16-byte stack alignment. Handle successful short writes;
    // zero/error from write terminates with status 1 (no silent success).
    fn print_runtime(&mut self) {
        self.prologue();
        self.emit(0xa9bf53f3); // stp x19, x20, [sp, #-16]!
        self.emit(0xaa0003f3); // mov x19, x0
        self.emit(0xaa0103f4); // mov x20, x1
        let done = self.new_label();
        let again = self.new_label();
        let error = self.new_label();
        self.branch(0xb4000014, 19, 5, Target::Local(done)); // cbz x20
        self.label(again);
        self.immediate(0, 1); // stdout
        self.emit(0xaa1303e1); // mov x1, x19
        self.emit(0xaa1403e2); // mov x2, x20
        self.import(0); // libSystem write
        self.emit(0xf100001f); // cmp x0, #0
        self.branch(0x5400000d, 19, 5, Target::Local(error)); // b.le
        self.emit(0x8b000273); // add x19, x19, x0
        self.emit(0xcb000294); // sub x20, x20, x0
        self.branch(0xb5000014, 19, 5, Target::Local(again)); // cbnz x20
        self.label(done);
        self.emit(0xa8c153f3); // ldp x19, x20, [sp], #16
        self.epilogue();
        self.label(error);
        self.immediate(0, 1);
        self.import(1); // libSystem exit(1)
        self.emit(0xd4200000); // brk if a supposedly noreturn call returns
    }
}

pub fn generate(program: &Program) -> Result<Code, String> {
    let mut e = Encoder::default();
    // LC_MAIN receives the Darwin C entry ABI. Return a successful int to dyld.
    e.prologue();
    e.call(Target::Function(program.main));
    e.immediate(0, 0);
    e.epilogue();

    let mut functions = Vec::new();
    for function in &program.functions {
        functions.push(e.position());
        e.prologue();
        for op in &function.ops {
            match op {
                Op::Write(value) => e.string(value),
                Op::Call(id) => e.call(Target::Function(*id)),
            }
        }
        e.epilogue();
    }
    let print = e.position();
    e.print_runtime();
    let instruction_bytes = e.words.len() * 4;
    let mut constants = Vec::new();
    let mut string_offsets = Vec::new();
    for value in &e.strings {
        string_offsets.push(CODE_OFFSET + instruction_bytes + constants.len());
        constants.extend_from_slice(value);
    }
    for fixup in e.fixups {
        match fixup {
            Fixup::Branch {
                at,
                opcode,
                bits,
                shift,
                target,
            } => {
                let target = match target {
                    Target::Function(id) => functions[id],
                    Target::Print => print,
                    Target::Local(id) => e.locals[id],
                };
                let delta = target as i64 - at as i64;
                let limit = 1_i64 << (bits - 1);
                if !(-limit..limit).contains(&delta) {
                    return Err("function is out of ARM64 branch range".into());
                }
                e.words[at] = opcode | (((delta as u32) & ((1 << bits) - 1)) << shift);
            }
            Fixup::String { at, index } => {
                let address = string_offsets[index];
                e.words[at] = adrp(0, CODE_OFFSET + at * 4, address)?;
                e.words[at + 1] = 0x91000000 | (((address & 4095) as u32) << 10);
            }
        }
    }
    let mut bytes: Vec<u8> = e.words.into_iter().flat_map(u32::to_le_bytes).collect();
    let constants_bytes = constants.len();
    bytes.extend(constants);
    Ok(Code {
        bytes,
        instruction_bytes,
        constants_bytes,
        imports: e.imports,
    })
}
