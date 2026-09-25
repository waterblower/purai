use growing_byte_model::{
    Result,
    baseline::Ngram,
    data,
    graph::{Model, Params},
    model_io,
    predict::{self, GenerateOptions, Predictor},
    train::{self, Options},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const HELP: &str = "Growing byte model: learned context branches, shared byte classes, MDL pruning

  cargo run train --input data.txt --output output.model
  cargo run infer --model output.model --prompt \"Once upon a time,\"

  cargo run train --input data/train.jsonl --validation data/validation.jsonl --output output.model
  cargo run eval --model output.model --input data/validation.jsonl
  cargo run inspect --model output.model
  cargo run benchmark --model output.model --input benchmarks/data/blimp/data --report blimp.report.json

train:
  --validation PATH       independent validation data; auto-detects sibling validation.jsonl
  --valid-fraction 0.1     otherwise hold out the tail of the input
  --calibration-fraction 0.05  training-side held-out portion for copy reliability (0 disables)
  --rounds 12 --max-depth 12 --max-nodes 60000
  --candidates 100000 --forks 10000 --merges 64 --merge-top 16
  --min-count 4 --structure-weight 0.25 --discount 0.75 --theta 2
  --baseline-order 5      -1 skips the fixed n-gram comparison
  --report PATH           defaults to OUTPUT with .report.json extension
  --no-match              skip copy calibration and use the context graph alone
infer:
  --n 400 --temperature 0.85 --top-p 1 --seed 42
  --max-steps N            cap context traversal per byte; does not cap match scanning
  --no-match --stats --trace
  --prompt-file PATH       raw-byte prompt, instead of --prompt
  --output PATH            save continuation bytes exactly, instead of displaying prompt + continuation
eval:
  --no-match --max-steps N --report PATH
inspect:
  --top 20
benchmark:
  --reference PATH        BLiMP authors' models_summary.jsonl (optional)
  --samples PATH          write per-pair scores as JSONL (optional)
  --allow-partial         allow a diagnostic subset; default requires all 67 x 1000 pairs
  Graph-only BLiMP evaluation; matching/copying is always disabled.

.jsonl inputs contain JSON strings or objects with a string 'text' field;
other inputs are raw bytes. Each JSONL record starts a fresh history.
";

struct Args {
    values: BTreeMap<String, String>,
    flags: BTreeSet<String>,
}
impl Args {
    fn parse(words: &[String], value_names: &[&str], flag_names: &[&str]) -> Result<Self> {
        let mut a = Self {
            values: BTreeMap::new(),
            flags: BTreeSet::new(),
        };
        let mut i = 0;
        while i < words.len() {
            let word = words[i]
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument {}", words[i]))?;
            let (name, inline) = word
                .split_once('=')
                .map_or((word, None), |(k, v)| (k, Some(v)));
            if flag_names.contains(&name) {
                if inline.is_some() || !a.flags.insert(name.to_owned()) {
                    return Err(format!("invalid or repeated flag --{name}").into());
                }
            } else if value_names.contains(&name) {
                let value = if let Some(v) = inline {
                    v.to_owned()
                } else {
                    i += 1;
                    words
                        .get(i)
                        .filter(|v| !v.starts_with("--"))
                        .ok_or_else(|| format!("missing value for --{name}"))?
                        .clone()
                };
                if a.values.insert(name.to_owned(), value).is_some() {
                    return Err(format!("repeated option --{name}").into());
                }
            } else {
                return Err(format!("unknown option --{name}; use --help").into());
            }
            i += 1;
        }
        Ok(a)
    }
    fn required(&self, name: &str) -> Result<&str> {
        self.values
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("missing --{name}").into())
    }
    fn string(&self, name: &str, default: &str) -> String {
        self.values
            .get(name)
            .cloned()
            .unwrap_or_else(|| default.to_owned())
    }
    fn number<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T> {
        self.values.get(name).map_or(Ok(default), |v| {
            v.parse()
                .map_err(|_| format!("invalid --{name}: {v}").into())
        })
    }
    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }
}

fn evaluation_json(e: &predict::Evaluation) -> Value {
    json!({"bytes":e.bytes,"bits":e.bits,"bits_per_byte":e.bpb(),"context_steps_per_byte":e.steps_per_byte(),
        "min_context_steps":e.min_steps,"max_context_steps":e.max_steps,"match_predictions":e.match_predictions})
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    fs::write(path, text)?;
    Ok(())
}
fn same_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn train_command(words: &[String]) -> Result<()> {
    let a = Args::parse(
        words,
        &[
            "input",
            "output",
            "validation",
            "valid-fraction",
            "calibration-fraction",
            "rounds",
            "max-depth",
            "max-nodes",
            "candidates",
            "forks",
            "merges",
            "merge-top",
            "min-count",
            "structure-weight",
            "discount",
            "theta",
            "report",
            "baseline-order",
        ],
        &["no-match"],
    )?;
    let input = Path::new(a.required("input")?);
    let output = Path::new(a.required("output")?);
    if same_path(input, output) {
        return Err("output must not overwrite training input".into());
    }
    let validation_path = a
        .values
        .get("validation")
        .map(PathBuf::from)
        .or_else(|| data::default_validation(input));
    if validation_path
        .as_ref()
        .is_some_and(|p| same_path(p, input) || same_path(p, output))
    {
        return Err("training, validation and model output must have different paths".into());
    }
    let source = data::read_docs(input)?;
    let (training, validation) = if let Some(path) = &validation_path {
        (source, data::read_docs(path)?)
    } else {
        data::split_tail(source, a.number("valid-fraction", 0.1f64)?)?
    };
    let calibration_fraction = a.number("calibration-fraction", 0.05f64)?;
    if !calibration_fraction.is_finite() || !(0.0..1.0).contains(&calibration_fraction) {
        return Err("calibration-fraction must be in [0, 1)".into());
    }
    let (training, calibration) = if calibration_fraction == 0.0 || a.flag("no-match") {
        (training, vec![])
    } else {
        data::split_tail(training, calibration_fraction)?
    };
    let mut options = Options::default();
    options.rounds = a.number("rounds", options.rounds)?;
    options.max_depth = a.number("max-depth", options.max_depth)?;
    options.max_nodes = a.number("max-nodes", options.max_nodes)?;
    options.max_candidates = a.number("candidates", options.max_candidates)?;
    options.forks_per_round = a.number("forks", options.forks_per_round)?;
    options.merges_per_round = a.number("merges", options.merges_per_round)?;
    options.merge_top = a.number("merge-top", options.merge_top)?;
    options.min_count = a.number("min-count", options.min_count)?;
    options.structure_weight = a.number("structure-weight", options.structure_weight)?;
    options.validate()?;
    let params = Params {
        discount: a.number("discount", 0.75)?,
        theta: a.number("theta", 2.0)?,
    };
    params.validate()?;
    let baseline_order: i32 = a.number("baseline-order", 5)?;
    if !(-1..=8).contains(&baseline_order) {
        return Err("baseline-order must be -1..8".into());
    }
    let report_path = a
        .values
        .get("report")
        .map(PathBuf::from)
        .unwrap_or_else(|| output.with_extension("report.json"));
    if [input, output].iter().any(|p| same_path(&report_path, p))
        || validation_path
            .as_ref()
            .is_some_and(|p| same_path(&report_path, p))
    {
        return Err("report must have a different path from data and model".into());
    }
    println!(
        "structure: {} docs / {} bytes; calibration: {} docs / {} bytes; validation: {} docs / {} bytes",
        training.len(),
        data::byte_count(&training),
        calibration.len(),
        data::byte_count(&calibration),
        validation.len(),
        data::byte_count(&validation)
    );
    println!(
        "round   nodes classes depth  candidates   +grow  merge prune  valid_graph_bpb  seconds"
    );
    let start = Instant::now();
    let mut rounds = Vec::new();
    let mut model = train::train(&training, params.clone(), &options, |m, r| {
        let eval = Predictor::new(m).evaluate(&validation, false, usize::MAX);
        println!(
            "{:5} {:7} {:7} {:5} {:11} {:7} {:6} {:5} {:16.5} {:8.2}",
            r.round,
            r.nodes,
            r.classes,
            r.depth,
            r.candidates,
            r.added,
            r.merged,
            r.pruned,
            eval.bpb(),
            r.seconds
        );
        rounds.push(json!({"round":r.round,"nodes":r.nodes,"classes":r.classes,"depth":r.depth,
            "candidates":r.candidates,"added":r.added,"merged":r.merged,"pruned":r.pruned,
            "estimated_local_gain_bits":r.estimated_gain_bits,"seconds":r.seconds,"valid_graph_bpb":eval.bpb()}));
    })?;
    predict::calibrate(&mut model, &calibration);
    let train_seconds = start.elapsed().as_secs_f64();
    let size = model_io::save(output, &model)?;
    // Exercise the actual serialized artifact before reporting its metrics.
    let loaded = model_io::load(output)?;
    if model != loaded {
        return Err("saved model differs from trained model".into());
    }
    let predictor = Predictor::new(&loaded);
    let eval_start = Instant::now();
    let with_match = predictor.evaluate(&validation, true, usize::MAX);
    let eval_seconds = eval_start.elapsed().as_secs_f64();
    let graph_only = predictor.evaluate(&validation, false, usize::MAX);
    let budget4 = predictor.evaluate(&validation, false, 4);
    let root = predictor.evaluate(&validation, false, 1);
    println!(
        "final: with_match={:.5} graph_only={:.5} root_only={:.5} budget4={:.5} bits/byte",
        with_match.bpb(),
        graph_only.bpb(),
        root.bpb(),
        budget4.bpb()
    );
    println!(
        "context steps/byte={:.2} min={} max={} (slot tests + root access, not CPU instructions)",
        with_match.steps_per_byte(),
        with_match.min_steps.unwrap_or(0),
        with_match.max_steps
    );
    println!(
        "saved {} ({} bytes, {} nodes, {} classes); training {:.2}s",
        output.display(),
        size,
        model.nodes.len(),
        model.classes(),
        train_seconds
    );
    let baseline = if baseline_order >= 0 {
        let baseline_start = Instant::now();
        let ngram = Ngram::fit(&training, baseline_order as usize, params.clone());
        let bits = ngram.evaluate(&validation);
        let bpb = bits / data::byte_count(&validation) as f64;
        println!(
            "fixed order-{baseline_order} n-gram: {bpb:.5} bits/byte, {} contexts",
            ngram.contexts()
        );
        json!({"order":baseline_order,"contexts":ngram.contexts(),"bits_per_byte":bpb,"seconds":baseline_start.elapsed().as_secs_f64()})
    } else {
        Value::Null
    };
    let report = json!({"format":"growing-byte-model experiment v1","input":input,"validation":validation_path,
        "structure_documents":training.len(),"structure_bytes":data::byte_count(&training),
        "calibration_documents":calibration.len(),"calibration_bytes":data::byte_count(&calibration),
        "validation_documents":validation.len(),"validation_bytes":data::byte_count(&validation),
        "parameters":{"discount":params.discount,"theta":params.theta},
        "options":{"rounds":options.rounds,"max_depth":options.max_depth,"max_nodes":options.max_nodes,
            "max_candidates":options.max_candidates,"forks_per_round":options.forks_per_round,"merges_per_round":options.merges_per_round,
            "merge_top":options.merge_top,"min_count":options.min_count,"structure_weight":options.structure_weight},
        "rounds":rounds,"model_bytes":size,"nodes":model.nodes.len(),"classes":model.classes(),"max_depth":model.max_depth(),
        "training_seconds":train_seconds,"evaluation_seconds":eval_seconds,
        "with_match":evaluation_json(&with_match),"graph_only":evaluation_json(&graph_only),
        "root_only":evaluation_json(&root),"graph_budget4":evaluation_json(&budget4),"ngram_baseline":baseline,
        "protocol":"Document boundaries reset all history. Calibration is withheld from the training input. Validation is reporting-only: it does not choose edits, epochs, parameters or calibration. Metrics are byte-weighted. No validation-based best-checkpoint selection."});
    write_json(&report_path, &report)?;
    println!("report: {}", report_path.display());
    Ok(())
}

fn render_byte(b: u8) -> String {
    match b {
        b'\n' => "\\n".into(),
        b'\t' => "\\t".into(),
        b' ' => "␣".into(),
        33..=126 => (b as char).to_string(),
        _ => format!("\\x{b:02x}"),
    }
}
fn context_name(model: &Model, mut id: usize) -> String {
    if id == 0 {
        return "<root>".into();
    }
    let mut parts = Vec::new();
    while let Some(p) = model.nodes[id].parent {
        let label = &model.nodes[id].label;
        let text = label.bytes().map(render_byte).collect::<Vec<_>>().join("|");
        parts.push(if label.len() > 1 {
            format!("[{{{text}}}]")
        } else {
            format!("[{text}]")
        });
        id = p;
    }
    parts.join("")
}

fn infer_command(words: &[String]) -> Result<()> {
    let a = Args::parse(
        words,
        &[
            "model",
            "prompt",
            "prompt-file",
            "n",
            "temperature",
            "top-p",
            "seed",
            "max-steps",
            "output",
        ],
        &["no-match", "stats", "trace"],
    )?;
    if a.values.contains_key("prompt") && a.values.contains_key("prompt-file") {
        return Err("choose --prompt or --prompt-file".into());
    }
    let model_path = Path::new(a.required("model")?);
    if a.values.get("output").is_some_and(|output| {
        same_path(Path::new(output), model_path)
            || a.values
                .get("prompt-file")
                .is_some_and(|p| same_path(Path::new(output), Path::new(p)))
    }) {
        return Err("output must not overwrite model or prompt file".into());
    }
    let model = model_io::load(model_path)?;
    let predictor = Predictor::new(&model);
    let prompt = if let Some(path) = a.values.get("prompt-file") {
        fs::read(path)?
    } else {
        a.string("prompt", "").into_bytes()
    };
    let opts = GenerateOptions {
        bytes: a.number("n", 400)?,
        temperature: a.number("temperature", 0.85)?,
        top_p: a.number("top-p", 1.0)?,
        seed: a.number("seed", 42)?,
        use_match: !a.flag("no-match"),
        budget: a.number("max-steps", usize::MAX)?,
    };
    if opts.bytes > 10_000_000 {
        return Err("--n exceeds the 10000000-byte generation limit".into());
    }
    let mut steps = 0usize;
    let mut min_steps = usize::MAX;
    let mut max_steps = 0;
    let mut matches = 0;
    let continuation = predictor.generate(&prompt, &opts, |b, st, p| {
        steps += st.steps;
        min_steps = min_steps.min(st.steps);
        max_steps = max_steps.max(st.steps);
        if st.match_byte.is_some() {
            matches += 1;
        }
        if a.flag("trace") {
            eprintln!(
                "{} p={:.5} steps={} context={} evidence={} match_len={}",
                render_byte(b),
                p[b as usize],
                st.steps,
                context_name(&model, st.node),
                model.nodes[st.node].counts.total(),
                st.match_length
            );
        }
    })?;
    if let Some(path) = a.values.get("output") {
        fs::write(path, &continuation)?;
        eprintln!("wrote {} continuation bytes to {path}", continuation.len());
    } else {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        out.write_all(&prompt)?;
        out.write_all(&continuation)?;
        out.write_all(b"\n")?;
    }
    if a.flag("stats") {
        eprintln!(
            "generated={} bytes; context steps/byte={:.2} min={} max={}; match proposals={}",
            continuation.len(),
            steps as f64 / continuation.len().max(1) as f64,
            if continuation.is_empty() {
                0
            } else {
                min_steps
            },
            max_steps,
            matches
        );
    }
    Ok(())
}

fn eval_command(words: &[String]) -> Result<()> {
    let a = Args::parse(
        words,
        &["model", "input", "max-steps", "report"],
        &["no-match"],
    )?;
    let budget = a.number("max-steps", usize::MAX)?;
    if budget == 0 {
        return Err("max-steps must be positive".into());
    }
    let model = model_io::load(a.required("model")?)?;
    let docs = data::read_docs(a.required("input")?)?;
    let predictor = Predictor::new(&model);
    let start = Instant::now();
    let e = predictor.evaluate(&docs, !a.flag("no-match"), budget);
    println!(
        "bits/byte={:.6} bytes={} docs={} context_steps/byte={:.2} seconds={:.3}",
        e.bpb(),
        e.bytes,
        docs.len(),
        e.steps_per_byte(),
        start.elapsed().as_secs_f64()
    );
    if let Some(path) = a.values.get("report") {
        if [a.required("model")?, a.required("input")?]
            .iter()
            .any(|p| same_path(Path::new(path), Path::new(p)))
        {
            return Err("report must not overwrite model/input".into());
        }
        write_json(Path::new(path), &evaluation_json(&e))?;
    }
    Ok(())
}

fn benchmark_command(words: &[String]) -> Result<()> {
    let a = Args::parse(
        words,
        &["model", "input", "report", "samples", "reference"],
        &["allow-partial"],
    )?;
    let input = Path::new(a.required("input")?);
    let report_path = Path::new(a.required("report")?);
    let inputs = [
        Some(a.required("model")?),
        a.values.get("reference").map(String::as_str),
    ];
    let outputs = [Some(report_path), a.values.get("samples").map(Path::new)];
    for path in outputs.iter().flatten() {
        if inputs
            .iter()
            .flatten()
            .any(|p| same_path(path, Path::new(p)))
            || path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."))
                .canonicalize()?
                .starts_with(input.canonicalize()?)
        {
            return Err("benchmark output must not overwrite model/reference or be inside the input directory".into());
        }
    }
    if outputs[1].is_some_and(|p| same_path(report_path, p)) {
        return Err("report and samples must have different paths".into());
    }
    let start = Instant::now();
    let model = model_io::load(a.required("model")?)?;
    let predictor = Predictor::new(&model);
    let mut sink = outputs[1]
        .map(fs::File::create)
        .transpose()?
        .map(io::BufWriter::new);
    let mut report = growing_byte_model::benchmark::blimp(
        &predictor,
        input,
        sink.as_mut().map(|s| s as &mut dyn Write),
    )?;
    if let Some(sink) = &mut sink {
        sink.flush()?;
    }
    if report["full_suite"] != true && !a.flag("allow-partial") {
        return Err("expected 67 paradigms with pairIDs 0..999 each; use --allow-partial for a diagnostic subset".into());
    }
    if let Some(reference) = a.values.get("reference") {
        growing_byte_model::benchmark::add_references(&mut report, Path::new(reference))?;
    }
    report["model"] = json!({"path":a.required("model")?,"bytes":fs::metadata(a.required("model")?)?.len(),
        "nodes":model.nodes.len(),"max_depth":model.max_depth(),"training_bytes":model.training_bytes});
    report["input"] = json!(input);
    report["seconds_including_load"] = json!(start.elapsed().as_secs_f64());
    write_json(report_path, &report)?;
    println!(
        "BLiMP: accuracy={:.4}% pairs={} full_suite={} seconds={:.3}",
        100.0 * report["overall"]["accuracy"].as_f64().unwrap(),
        report["overall"]["pairs"],
        report["full_suite"],
        start.elapsed().as_secs_f64()
    );
    println!("report: {}", report_path.display());
    Ok(())
}

fn inspect_command(words: &[String]) -> Result<()> {
    let a = Args::parse(words, &["model", "top"], &[])?;
    let model = model_io::load(a.required("model")?)?;
    println!(
        "nodes={} classes={} max_depth={} structure_bytes={} calibration_bytes={} discount={} theta={}",
        model.nodes.len(),
        model.classes(),
        model.max_depth(),
        model.training_bytes,
        model.calibration_bytes,
        model.params.discount,
        model.params.theta
    );
    let mut ids: Vec<usize> = (1..model.nodes.len()).collect();
    ids.sort_unstable_by(|&a, &b| {
        model.nodes[b]
            .depth
            .cmp(&model.nodes[a].depth)
            .then(
                model.nodes[b]
                    .counts
                    .total()
                    .cmp(&model.nodes[a].counts.total()),
            )
            .then(a.cmp(&b))
    });
    for id in ids.into_iter().take(a.number("top", 20usize)?) {
        let n = &model.nodes[id];
        let mut top = n.counts.0.clone();
        top.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!(
            "#{id} parent={} depth={} N={} {} -> {}",
            n.parent.unwrap(),
            n.depth,
            n.counts.total(),
            context_name(&model, id),
            top.into_iter()
                .take(4)
                .map(|(b, c)| format!("{}:{c}", render_byte(b)))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!("learned shared byte classes:");
    for (id, n) in model
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.label.len() > 1)
        .take(a.number("top", 20usize)?)
    {
        println!("#{id} N={} {}", n.counts.total(), context_name(&model, id));
    }
    Ok(())
}

fn run() -> Result<()> {
    let words: Vec<String> = std::env::args().skip(1).collect();
    if words.is_empty() || words.iter().any(|s| s == "--help" || s == "-h") {
        print!("{HELP}");
        return Ok(());
    }
    match words[0].as_str() {
        "train" => train_command(&words[1..]),
        "infer" => infer_command(&words[1..]),
        "eval" => eval_command(&words[1..]),
        "benchmark" => benchmark_command(&words[1..]),
        "inspect" => inspect_command(&words[1..]),
        other => Err(format!("unknown command {other}; use --help").into()),
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
