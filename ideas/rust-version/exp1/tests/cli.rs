use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn bpb_increase_stops_cli_and_saves_the_previous_round() {
    let dir = std::env::temp_dir().join(format!("bpb-stop-test-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let input = dir.join("train.txt");
    let validation = dir.join("valid.txt");
    fs::write(&input, "abx cby\n".repeat(80)).unwrap();
    // The second-order training pattern contradicts the held-out continuation.
    fs::write(&validation, "aby cbx\n").unwrap();
    let model = dir.join("stopped.model");
    let first = dir.join("first.model");
    for (output, rounds, stop) in [(&model, "36", true), (&first, "1", false)] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_growing-byte-model"));
        cmd.args(["train", "--input"])
            .arg(&input)
            .arg("--validation")
            .arg(&validation)
            .arg("--output")
            .arg(output)
            .args(["--rounds", rounds, "--baseline-order", "-1", "--no-match"]);
        if stop {
            cmd.arg("--stop-on-bpb-increase");
        }
        let result = cmd.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(model.with_extension("report.json")).unwrap()).unwrap();
    assert_eq!(report["stopping"]["reason"], "validation_bpb_increased");
    assert_eq!(report["stopping"]["completed_rounds"], 2);
    assert_eq!(report["stopping"]["selected_round"], 1);
    assert_eq!(report["rounds"].as_array().unwrap().len(), 2);
    assert_eq!(
        report["graph_only"]["bits_per_byte"],
        report["rounds"][0]["valid_graph_bpb"]
    );
    assert_eq!(fs::read(model).unwrap(), fs::read(first).unwrap());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn requested_train_infer_workflow_and_input_errors() {
    let dir = std::env::temp_dir().join(format!(
        "growing-byte-test-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    let input = dir.join("data.txt");
    let model = dir.join("output.model");
    let generated = dir.join("generated.bin");
    fs::write(
        &input,
        "Once upon a time, a little girl found a red ball.\n".repeat(80),
    )
    .unwrap();
    let exe = env!("CARGO_BIN_EXE_growing-byte-model");
    let train = Command::new(exe)
        .args(["train", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&model)
        .args(["--rounds", "4", "--baseline-order", "-1"])
        .output()
        .unwrap();
    assert!(
        train.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&train.stdout),
        String::from_utf8_lossy(&train.stderr)
    );
    assert!(model.exists());
    assert!(model.with_extension("report.json").exists());
    let infer = Command::new(exe)
        .args(["infer", "--model"])
        .arg(&model)
        .args(["--prompt", "Once upon a time,", "--n", "50", "--output"])
        .arg(&generated)
        .output()
        .unwrap();
    assert!(
        infer.status.success(),
        "{}",
        String::from_utf8_lossy(&infer.stderr)
    );
    assert_eq!(fs::read(&generated).unwrap().len(), 50);
    let eval = Command::new(exe)
        .args(["eval", "--model"])
        .arg(&model)
        .arg("--input")
        .arg(&input)
        .output()
        .unwrap();
    assert!(eval.status.success());
    let bad = Command::new(exe)
        .args(["infer", "--model"])
        .arg(&model)
        .args(["--temperature", "NaN"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    let original = fs::read(&input).unwrap();
    let bad = Command::new(exe)
        .args(["train", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&input)
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert_eq!(original, fs::read(&input).unwrap());
    let jsonl = dir.join("bad.jsonl");
    fs::write(&jsonl, "{\"not_text\":3}\n").unwrap();
    let bad = Command::new(exe)
        .args(["train", "--input"])
        .arg(&jsonl)
        .arg("--output")
        .arg(&model)
        .output()
        .unwrap();
    assert!(!bad.status.success());
    // Reporting a different validation corpus must not change the learned model.
    let validation_a = dir.join("validation-a.txt");
    let validation_b = dir.join("validation-b.txt");
    fs::write(&validation_a, b"a completely different held out story").unwrap();
    fs::write(&validation_b, b"\x00\xff\x01 unrelated validation bytes").unwrap();
    let model_a = dir.join("a.model");
    let model_b = dir.join("b.model");
    for (validation, output) in [(&validation_a, &model_a), (&validation_b, &model_b)] {
        let run = Command::new(exe)
            .args(["train", "--input"])
            .arg(&input)
            .arg("--validation")
            .arg(validation)
            .arg("--output")
            .arg(output)
            .args(["--rounds", "3", "--baseline-order", "-1"])
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
    assert_eq!(fs::read(&model_a).unwrap(), fs::read(&model_b).unwrap());
    fs::remove_dir_all(dir).unwrap();
}
