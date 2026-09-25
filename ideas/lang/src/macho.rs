//! An in-process Mach-O writer/linker for arm64 macOS. Format constants follow
//! Apple's mach-o/loader.h and xnu/osfmk/kern/cs_blobs.h. This writes dyld bind
//! opcodes and a SHA-256 ad-hoc signature itself; no SDK or linker is invoked.
use crate::aarch64::{CODE_OFFSET, Code};
use crate::sha256;

const PAGE: usize = 16384;
const BASE: u64 = 0x1_0000_0000;
const IDENTIFIER: &[u8] = b"lang.executable\0";

fn align(n: usize, alignment: usize) -> usize {
    (n + alignment - 1) & !(alignment - 1)
}
fn u32le(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn u64le(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn u32be(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_be_bytes());
}
fn u64be(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_be_bytes());
}
fn fixed_name(out: &mut Vec<u8>, name: &str) {
    assert!(name.len() <= 16);
    out.extend_from_slice(name.as_bytes());
    out.resize(out.len() + 16 - name.len(), 0);
}

struct Section<'a> {
    name: &'a str,
    offset: usize,
    size: usize,
    align: u32,
    flags: u32,
}

fn segment(
    out: &mut Vec<u8>,
    name: &str,
    offset: usize,
    size: usize,
    protection: u32,
    sections: &[Section<'_>],
) {
    let pagezero = name == "__PAGEZERO";
    u32le(out, 0x19); // LC_SEGMENT_64
    u32le(out, (72 + 80 * sections.len()) as u32);
    fixed_name(out, name);
    u64le(out, if pagezero { 0 } else { BASE + offset as u64 });
    u64le(
        out,
        if pagezero {
            BASE
        } else {
            align(size, PAGE) as u64
        },
    );
    u64le(out, offset as u64);
    u64le(out, size as u64);
    u32le(out, protection); // maxprot
    u32le(out, protection); // initprot
    u32le(out, sections.len() as u32);
    u32le(out, 0);
    for section in sections {
        fixed_name(out, section.name);
        fixed_name(out, name);
        u64le(out, BASE + section.offset as u64);
        u64le(out, section.size as u64);
        u32le(out, section.offset as u32);
        u32le(out, section.align);
        u32le(out, 0); // reloff
        u32le(out, 0); // nreloc
        u32le(out, section.flags);
        u32le(out, 0); // reserved1 (indirect-symbol-table index for __got)
        u32le(out, 0);
        u32le(out, 0);
    }
}

fn path_command(out: &mut Vec<u8>, command: u32, path: &str, dylib: bool) {
    let header = if dylib { 24 } else { 12 };
    let size = align(header + path.len() + 1, 8);
    let start = out.len();
    u32le(out, command);
    u32le(out, size as u32);
    u32le(out, header as u32);
    if dylib {
        u32le(out, 0); // timestamp
        u32le(out, 0x10000); // current version
        u32le(out, 0x10000); // compatibility version
    }
    out.extend_from_slice(path.as_bytes());
    out.resize(start + size, 0);
}

fn signature_size(limit: usize) -> usize {
    20 + 88 + IDENTIFIER.len() + limit.div_ceil(4096) * 32
}

fn signature(image: &[u8], text_size: usize) -> Vec<u8> {
    let total = signature_size(image.len());
    let mut out = Vec::with_capacity(total);
    // SuperBlob with a single CodeDirectory entry. Signing structs are big-endian.
    for n in [0xfade0cc0, total as u32, 1, 0, 20] {
        u32be(&mut out, n);
    }
    for n in [
        0xfade0c02,
        (total - 20) as u32,
        0x20400,
        0x20002,
        (88 + IDENTIFIER.len()) as u32,
        88,
        0,
        image.len().div_ceil(4096) as u32,
        image.len() as u32,
    ] {
        u32be(&mut out, n);
    }
    out.extend_from_slice(&[32, 2, 0, 12]); // SHA-256, 4 KiB hash pages
    for _ in 0..4 {
        u32be(&mut out, 0);
    }
    u64be(&mut out, 0); // codeLimit64 unused for these <4 GiB images
    u64be(&mut out, 0); // executable segment file offset
    u64be(&mut out, text_size as u64);
    u64be(&mut out, 1); // CS_EXECSEG_MAIN_BINARY
    debug_assert_eq!(out.len(), 20 + 88);
    out.extend_from_slice(IDENTIFIER);
    for page in image.chunks(4096) {
        out.extend_from_slice(&sha256::digest(page));
    }
    debug_assert_eq!(out.len(), total);
    out
}

pub fn executable(mut code: Code) -> Result<Vec<u8>, String> {
    let text_size = align(CODE_OFFSET + code.bytes.len(), PAGE);
    let data_offset = text_size;
    let linkedit_offset = data_offset + PAGE;
    if linkedit_offset > u32::MAX as usize / 2 {
        return Err("executable exceeds the prototype's 2 GiB limit".into());
    }
    code.bind_imports(data_offset)?;

    // Bind __DATA's two pointer slots to libSystem's write and exit symbols.
    // SET_DYLIB_ORDINAL(1), SET_TYPE(pointer), SET_SEGMENT(2, offset=0).
    let mut linkedit = vec![0x11, 0x51, 0x72, 0x00];
    for symbol in [b"_write\0".as_slice(), b"_exit\0".as_slice()] {
        linkedit.push(0x40); // SET_SYMBOL_TRAILING_FLAGS
        linkedit.extend_from_slice(symbol);
        linkedit.push(0x90); // DO_BIND, advances the address by sizeof(pointer)
    }
    linkedit.push(0); // DONE
    let bind_size = linkedit.len();
    linkedit.resize(align(linkedit.len(), 8), 0);

    // Also provide ordinary nlist entries and an indirect symbol table so
    // platform inspection/debugging tools can understand the imported pointers.
    let symoff = linkedit_offset + linkedit.len();
    for strx in [1, 8] {
        u32le(&mut linkedit, strx);
        linkedit.extend_from_slice(&[1, 0]); // N_UNDF | N_EXT, NO_SECT
        linkedit.extend_from_slice(&0x100_u16.to_le_bytes()); // dylib ordinal 1
        u64le(&mut linkedit, 0);
    }
    let stroff = linkedit_offset + linkedit.len();
    let strings = b"\0_write\0_exit\0";
    linkedit.extend_from_slice(strings);
    linkedit.resize(align(linkedit.len(), 4), 0);
    let indirectoff = linkedit_offset + linkedit.len();
    u32le(&mut linkedit, 0);
    u32le(&mut linkedit, 1);
    linkedit.resize(align(linkedit.len(), 16), 0);
    let sigoff = linkedit_offset + linkedit.len();
    let sigsize = signature_size(sigoff);
    let linkedit_size = linkedit.len() + sigsize;

    let mut commands = Vec::new();
    segment(&mut commands, "__PAGEZERO", 0, 0, 0, &[]);
    segment(
        &mut commands,
        "__TEXT",
        0,
        text_size,
        5,
        &[
            Section {
                name: "__text",
                offset: CODE_OFFSET,
                size: code.instruction_bytes,
                align: 2,
                flags: 0x80000400,
            },
            Section {
                name: "__const",
                offset: CODE_OFFSET + code.instruction_bytes,
                size: code.constants_bytes,
                align: 0,
                flags: 0,
            },
        ],
    );
    segment(
        &mut commands,
        "__DATA",
        data_offset,
        PAGE,
        3,
        &[Section {
            name: "__got",
            offset: data_offset,
            size: 16,
            align: 3,
            flags: 6,
        }],
    );
    segment(
        &mut commands,
        "__LINKEDIT",
        linkedit_offset,
        linkedit_size,
        1,
        &[],
    );

    u32le(&mut commands, 0x80000022); // LC_DYLD_INFO_ONLY
    u32le(&mut commands, 48);
    for n in [
        0,
        0,
        linkedit_offset as u32,
        bind_size as u32,
        0,
        0,
        0,
        0,
        0,
        0,
    ] {
        u32le(&mut commands, n);
    }

    for n in [2, 24, symoff as u32, 2, stroff as u32, strings.len() as u32] {
        u32le(&mut commands, n); // LC_SYMTAB
    }
    u32le(&mut commands, 0xb); // LC_DYSYMTAB
    u32le(&mut commands, 80);
    for n in [
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        indirectoff as u32,
        2,
        0,
        0,
        0,
        0,
    ] {
        u32le(&mut commands, n);
    }
    path_command(&mut commands, 0xe, "/usr/lib/dyld", false);
    path_command(&mut commands, 0xc, "/usr/lib/libSystem.B.dylib", true);
    // macOS 11 ABI, no SDK dependency. Avoid claiming a newer SDK's policies.
    for n in [0x32, 24, 1, 11 << 16, 11 << 16, 0] {
        u32le(&mut commands, n);
    }
    u32le(&mut commands, 0x80000028); // LC_MAIN
    u32le(&mut commands, 24);
    u64le(&mut commands, CODE_OFFSET as u64);
    u64le(&mut commands, 0);
    for n in [0x1d, 16, sigoff as u32, sigsize as u32] {
        u32le(&mut commands, n);
    }

    let mut image = Vec::new();
    for n in [
        0xfeedfacf,
        0x0100000c,
        0,
        2,
        12,
        commands.len() as u32,
        0x00200085, // MH_PIE | MH_TWOLEVEL | MH_DYLDLINK | MH_NOUNDEFS
        0,
    ] {
        u32le(&mut image, n);
    }
    image.extend(commands);
    if image.len() > CODE_OFFSET {
        return Err("Mach-O load commands exceed header space".into());
    }
    image.resize(CODE_OFFSET, 0);
    image.extend(code.bytes);
    image.resize(linkedit_offset, 0); // text padding and zeroed GOT/data segment
    image.extend(linkedit);
    let signed = signature(&image, text_size);
    image.extend(signed);
    Ok(image)
}
