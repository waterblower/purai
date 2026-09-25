use crate::{
    Result,
    graph::{ByteSet, Calibration, Histogram, Model, Node, Params},
};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"GBGRAPH\n";
const VERSION: u32 = 1;
const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_NODES: usize = 250_000;

/// FNV-1a detects accidental corruption; it is not authentication.
pub fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x100_0000_01b3)
    })
}
fn u32_out(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn u64_out(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}

/// Deterministic little-endian format. Derived routing/probability caches and
/// document-local inference state are never serialized.
pub fn encode(model: &Model) -> Result<Vec<u8>> {
    model.validate()?;
    if model.nodes.len() > MAX_NODES {
        return Err("model exceeds node limit".into());
    }
    let mut out = MAGIC.to_vec();
    u32_out(&mut out, VERSION);
    u64_out(&mut out, model.params.discount.to_bits());
    u64_out(&mut out, model.params.theta.to_bits());
    u64_out(&mut out, model.training_bytes);
    u64_out(&mut out, model.calibration_bytes);
    for table in [&model.calibration.hits, &model.calibration.totals] {
        for &n in table {
            u64_out(&mut out, n);
        }
    }
    u32_out(&mut out, model.nodes.len() as u32);
    for node in &model.nodes {
        u32_out(&mut out, node.parent.map_or(u32::MAX, |p| p as u32));
        for n in node.label.0 {
            u64_out(&mut out, n);
        }
        out.extend_from_slice(&(node.counts.0.len() as u16).to_le_bytes());
        for &(b, n) in &node.counts.0 {
            out.push(b);
            u64_out(&mut out, n);
        }
    }
    let sum = checksum(&out);
    u64_out(&mut out, sum);
    Ok(out)
}

struct Reader<'a> {
    data: &'a [u8],
    position: usize,
}
impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N)
            .ok_or("model length overflow")?;
        let bytes = self.data.get(self.position..end).ok_or("truncated model")?;
        self.position = end;
        Ok(bytes.try_into().unwrap())
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
}

pub fn decode(data: &[u8]) -> Result<Model> {
    if data.len() < 24 || data.len() as u64 > MAX_FILE_BYTES || &data[..8] != MAGIC {
        return Err("not a supported byte-graph model".into());
    }
    let (body, tail) = data.split_at(data.len() - 8);
    if checksum(body) != u64::from_le_bytes(tail.try_into().unwrap()) {
        return Err("model checksum mismatch".into());
    }
    let mut r = Reader {
        data: body,
        position: 8,
    };
    if r.u32()? != VERSION {
        return Err("unsupported model version".into());
    }
    let discount = f64::from_bits(r.u64()?);
    let theta = f64::from_bits(r.u64()?);
    let params = Params { discount, theta };
    params.validate()?;
    let training_bytes = r.u64()?;
    let calibration_bytes = r.u64()?;
    let mut calibration = Calibration::default();
    for table in [&mut calibration.hits, &mut calibration.totals] {
        for n in table {
            *n = r.u64()?;
        }
    }
    let count = r.u32()? as usize;
    if count == 0 || count > MAX_NODES || count > (body.len() - r.position) / 38 {
        return Err("invalid model node count".into());
    }
    let mut nodes: Vec<Node> = Vec::with_capacity(count);
    for id in 0..count {
        let raw_parent = r.u32()?;
        let parent = if id == 0 {
            if raw_parent != u32::MAX {
                return Err("invalid root parent".into());
            }
            None
        } else {
            let p = raw_parent as usize;
            if p >= id {
                return Err("forward/cyclic parent reference".into());
            }
            Some(p)
        };
        let mut label = ByteSet::default();
        for w in &mut label.0 {
            *w = r.u64()?;
        }
        let nonzero = u16::from_le_bytes(r.bytes()?) as usize;
        if nonzero > 256 {
            return Err("invalid histogram size".into());
        }
        let mut counts = Histogram(Vec::with_capacity(nonzero));
        let mut total = 0u64;
        for _ in 0..nonzero {
            let b = r.bytes::<1>()?[0];
            let c = r.u64()?;
            total = total.checked_add(c).ok_or("histogram overflow")?;
            counts.0.push((b, c));
        }
        if total > training_bytes {
            return Err("count exceeds training bytes".into());
        }
        let depth = parent.map_or(0, |p| nodes[p].depth + 1);
        nodes.push(Node {
            parent,
            label,
            counts,
            children: Vec::new(),
            depth,
        });
        if let Some(p) = parent {
            nodes[p].children.push(id);
        }
    }
    if r.position != body.len() {
        return Err("trailing model data".into());
    }
    let model = Model {
        params,
        nodes,
        calibration,
        training_bytes,
        calibration_bytes,
    };
    model.validate()?;
    Ok(model)
}

pub fn load(path: impl AsRef<Path>) -> Result<Model> {
    let file = fs::File::open(path)?;
    if file.metadata()?.len() > MAX_FILE_BYTES {
        return Err("model file exceeds 512 MiB limit".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes)?;
    decode(&bytes)
}

/// Write beside the destination, sync, then rename; a failed write does not
/// destroy an existing model. The temporary name is exclusively created.
pub fn save(path: impl AsRef<Path>, model: &Model) -> Result<usize> {
    let path = path.as_ref();
    let bytes = encode(model)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or("model output must name a file")?
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.{}.model.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let written = (|| -> Result<()> {
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written?;
    Ok(bytes.len())
}
