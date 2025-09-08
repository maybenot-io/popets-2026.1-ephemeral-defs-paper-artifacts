use std::{
    collections::HashMap, fs::OpenOptions, io::Write, process::Command, sync::Mutex, time::Duration,
};

use anyhow::{bail, Result};
use console::Emoji;
use indicatif::ParallelProgressIterator;
use log::{debug, info, warn};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    config::Config, get_progress_style, get_trace_content, make_dataset_map, read_dataset,
};

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct EvalConfig {
    /// same as used in sim: for overhead stats
    pub base_dataset: String,
    /// custom trace length for overhead stats and attacks, defaults to 5000
    pub trace_length: Option<usize>,
    /// path to Deep Fingerprinting script, see repository for details
    pub df: Option<String>,
    /// path to optional DF pre-trained model
    pub df_model: Option<String>,
    /// path to Robust Fingerprinting script, see repository for details
    pub rf: Option<String>,
    /// path to optional RF pre-trained model
    pub rf_model: Option<String>,
    /// number of epochs to run the attacks for, defaults to 30
    pub epochs: Option<usize>,
    // the number of epochs for early stopping (patience window)
    pub patience: Option<usize>,
    /// path to output csv file, if set, will append to it
    pub csv: Option<String>,
    /// if set, do not run some sanity checks related to simulation when
    /// computing overheads (because they might not hold for when comparing to
    /// real-world datasets)
    pub real_world_dataset: Option<bool>,
    /// if set, run on the specified fold (default is fold 0)
    pub fold: Option<usize>,
    /// seed for deterministic evaluation
    pub seed: Option<u64>,
}

pub fn do_eval(cfg: &Config, input: String, fold: Option<usize>) -> Result<()> {
    if cfg.eval.is_none() {
        bail!("eval config is missing");
    }
    let cfg = cfg.eval.as_ref().unwrap();

    if cfg.df.is_none() {
        warn!("df is not set, skipping DF attack");
    }
    if cfg.rf.is_none() {
        warn!("rf is not set, skipping RF attack");
    }
    if let Some(seed) = cfg.seed {
        info!("deterministic, using seed {seed}");
    }
    let fold = fold.unwrap_or(cfg.fold.unwrap_or(0));
    if fold >= 10 {
        bail!("fold {} is out of range, should be between 0 and 9", fold);
    }
    info!("using fold {fold} for attacks");

    // if csv is set, check that it would be possible to write or append to it
    if let Some(csv) = &cfg.csv {
        if std::path::Path::new(&csv).exists() {
            if !std::path::Path::new(&csv).is_file() {
                bail!(format!("csv {} is not a file", csv));
            }
        } else {
            // create the file
            std::fs::File::create(csv)?;

            // write header
            let mut csv_line = String::new();
            csv_line.push_str("dataset,dataset_length,");

            let template = "X,";
            for stat in OVERHEAD_KEYS.iter() {
                csv_line.push_str(&template.replace('X', stat));
            }

            csv_line.push_str("rf,df\n");

            std::fs::write(csv, csv_line)?;
        }
    }

    // create a buffer for the csv line
    let mut csv_line = String::new();

    info!("reading base dataset from {}", cfg.base_dataset);
    let base_dataset_dir = std::path::Path::new(&cfg.base_dataset);
    if !base_dataset_dir.exists() {
        bail!("base dataset {} does not exist", base_dataset_dir.display());
    }
    if !base_dataset_dir.is_dir() {
        bail!(
            "base dataset {} is not a directory",
            base_dataset_dir.display()
        );
    }
    let base_dataset = read_dataset(base_dataset_dir);
    let base_map = make_dataset_map(&base_dataset);
    info!("read {} traces", base_dataset.len());

    info!("reading input dataset from {input}");
    let input_dataset_dir = std::path::Path::new(&input);
    if !input_dataset_dir.exists() {
        bail!("input dataset {} does not exist", input);
    }
    if !input_dataset_dir.is_dir() {
        bail!("input dataset {} is not a directory", input);
    }
    let input_dataset = read_dataset(input_dataset_dir);
    info!("read {} traces", input_dataset.len());

    // if csv is set, write first column, the input dataset
    if cfg.csv.is_some() {
        csv_line.push_str(&format!("{input},"));
        csv_line.push_str(&format!("{},", input_dataset.len()));
    }

    let max = if cfg.trace_length.unwrap_or(5000) == 0 {
        usize::MAX
    } else {
        cfg.trace_length.unwrap_or(5000)
    };

    compute_overheads(
        base_map,
        &input_dataset,
        max,
        &mut csv_line,
        cfg.real_world_dataset.unwrap_or(false),
    );

    // find the maximum class and sample number
    let n_classes = input_dataset
        .iter()
        .map(|(class, _, _)| class)
        .max()
        .unwrap()
        + 1;
    let n_samples = input_dataset
        .iter()
        .map(|(_, fname, _)| {
            fname
                .split('-')
                .next_back()
                .unwrap()
                .replace(".log", "")
                .parse::<usize>()
                .unwrap_or(0)
        })
        .max()
        .unwrap()
        + 1;
    info!("found {n_classes} classes and {n_samples} samples in {input}");

    run_attacks(
        cfg,
        input,
        max,
        n_classes,
        n_samples,
        fold,
        &mut csv_line,
        cfg.seed,
    );

    if let Some(csv) = &cfg.csv {
        csv_line.push('\n');
        // done, append to the csv file
        let mut file = OpenOptions::new().append(true).open(csv)?;
        file.write_all(csv_line.as_bytes())?;
    }

    Ok(())
}

const OVERHEAD_KEYS: [&str; 15] = [
    "base",
    "base sent",
    "base recv",
    "defended",
    "defended sent",
    "defended recv",
    "missing",
    "missing sent",
    "missing recv",
    "load",
    "load sent",
    "load recv",
    "delay",
    "time base",
    "time defended",
];

fn compute_overheads(
    base_map: HashMap<String, String>,
    input_dataset: &[(usize, String, String)],
    max: usize,
    csv_line: &mut String,
    real_world: bool,
) {
    let capacity = base_map.len();

    let stats = Mutex::new(HashMap::new());
    {
        let mut stats_map = stats.lock().unwrap();
        for key in OVERHEAD_KEYS.iter() {
            stats_map.insert(*key, Vec::with_capacity(capacity));
        }
    }

    info!("computing overheads (max trace length {max})...");
    println!();
    input_dataset
        .par_iter()
        .progress_with_style(get_progress_style())
        .for_each(|(class, fname, trace)| {
            // either base dataset has the fname, or we skip
            let base_key = format!("{class}+{fname}");
            let base_fname = match base_map.get(&base_key) {
                Some(base_fname) => base_fname,
                None => {
                    return;
                }
            };
            let trace = &get_trace_content(trace);

            let (defended, _) = get_trace_stats(trace, max, fname, 0);
            if defended.sent_normal == 0 && defended.recv_normal == 0 {
                warn!("no normal traffic in {fname}, skipping");
                return;
            }
            let base_duration_at_n = if real_world {
                0
            } else {
                (defended.sent_normal + defended.recv_normal) as usize
            };
            let base_trace = &get_trace_content(base_fname);
            let (base, base_dur) = get_trace_stats(
                base_trace,
                max,
                fname,
                base_duration_at_n,
            );

            if !real_world && trace.len() < max && base_trace.len() < max {
                // should only hold if both traces are shorter than max
                assert!(base.sent_normal >= defended.sent_normal);
                assert!(base.recv_normal >= defended.recv_normal);
            }
            // not true for some datasets, unfortunately
            //assert_eq!(base.sent_padding, 0);
            //assert_eq!(base.recv_padding, 0);

            let mut metrics: HashMap<&str, f64> = HashMap::new();

            metrics.insert("base sent", base.sent_normal.into());
            metrics.insert("base recv", base.recv_normal.into());

            metrics.insert(
                "defended sent",
                (defended.sent_normal + defended.sent_padding).into(),
            );
            metrics.insert(
                "defended recv",
                (defended.recv_normal + defended.recv_padding).into(),
            );

            metrics.insert(
                "missing sent",
                (base.sent_normal - defended.sent_normal).into(),
            );
            metrics.insert(
                "missing recv",
                (base.recv_normal - defended.recv_normal).into(),
            );
            if let Some(base_dur) = base_dur {
                metrics.insert("time base", base_dur.as_secs_f64());
                metrics.insert("time defended", defended.duration_to_last_normal.as_secs_f64());

            } else {
                warn!("defended stats {defended:?}");
                warn!("base stats {base:?}");
                panic!("missing base duration for class {class} fname {fname}, implies extra normal traffic after defense simulation");
            }

            // copy the data into the stats map
            {
                let mut stats_map = stats.lock().unwrap();
                for (k, v) in metrics.iter() {
                    stats_map.entry(k).and_modify(|e| e.push(*v));
                }
            }
        });

    // compute overheads over entire dataset
    let n = base_map.len() as f64;
    let metrics = stats.lock().unwrap();

    let base_sent = metrics.get("base sent").unwrap().iter().sum::<f64>() / n;
    let base_recv = metrics.get("base recv").unwrap().iter().sum::<f64>() / n;
    let base = (metrics.get("base sent").unwrap().iter().sum::<f64>()
        + metrics.get("base recv").unwrap().iter().sum::<f64>())
        / n;

    let defended_sent = metrics.get("defended sent").unwrap().iter().sum::<f64>() / n;
    let defended_recv = metrics.get("defended recv").unwrap().iter().sum::<f64>() / n;
    let defended = (metrics.get("defended sent").unwrap().iter().sum::<f64>()
        + metrics.get("defended recv").unwrap().iter().sum::<f64>())
        / n;

    let load_sent = (metrics.get("defended sent").unwrap().iter().sum::<f64>()
        / metrics.get("base sent").unwrap().iter().sum::<f64>())
        - 1.0;
    let load_recv = (metrics.get("defended recv").unwrap().iter().sum::<f64>()
        / metrics.get("base recv").unwrap().iter().sum::<f64>())
        - 1.0;
    let load = (metrics.get("defended sent").unwrap().iter().sum::<f64>()
        + metrics.get("defended recv").unwrap().iter().sum::<f64>())
        / (metrics.get("base sent").unwrap().iter().sum::<f64>()
            + metrics.get("base recv").unwrap().iter().sum::<f64>())
        - 1.0;

    let missing_sent = metrics.get("missing sent").unwrap().iter().sum::<f64>()
        / metrics.get("base sent").unwrap().iter().sum::<f64>();
    let missing_recv = metrics.get("missing recv").unwrap().iter().sum::<f64>()
        / metrics.get("base recv").unwrap().iter().sum::<f64>();
    let missing = (metrics.get("missing sent").unwrap().iter().sum::<f64>()
        + metrics.get("missing recv").unwrap().iter().sum::<f64>())
        / (metrics.get("base sent").unwrap().iter().sum::<f64>()
            + metrics.get("base recv").unwrap().iter().sum::<f64>());

    let time_base = metrics.get("time base").unwrap().iter().sum::<f64>() / n;
    let time_defended = metrics.get("time defended").unwrap().iter().sum::<f64>() / n;
    let delay = (metrics.get("time defended").unwrap().iter().sum::<f64>()
        / metrics.get("time base").unwrap().iter().sum::<f64>())
        - 1.0;

    info!("METRIC\t\t  AVERAGE");
    info!("=======================================");
    info!("{:<10}\t{:8.1} tunnel events", "base 🌐", base);
    csv_line.push_str(&format!("{base:.4},"));
    info!("{:<10}\t{:8.1}", "-sent", base_sent);
    csv_line.push_str(&format!("{base_sent:.4},"));
    info!("{:<10}\t{:8.1}", "-recv", base_recv);
    csv_line.push_str(&format!("{base_recv:.4},"));

    info!("");
    info!("{:<10}\t{:8.1} tunnel events", "defended 🛡️", defended);
    csv_line.push_str(&format!("{defended:.4},"));
    info!("{:<10}\t{:8.1}", "-sent", defended_sent);
    csv_line.push_str(&format!("{defended_sent:.4},"));
    info!("{:<10}\t{:8.1}", "-recv", defended_recv);
    csv_line.push_str(&format!("{defended_recv:.4},"));

    info!("");
    info!("{:<10}\t{:8.2} fraction", "missing ⚠️", missing);
    csv_line.push_str(&format!("{missing:.4},"));
    info!("{:<10}\t{:8.2}", "-sent", missing_sent);
    csv_line.push_str(&format!("{missing_sent:.4},"));
    info!("{:<10}\t{:8.2}", "-recv", missing_recv);
    csv_line.push_str(&format!("{missing_recv:.4},"));

    info!("");
    info!("{:<10}\t{:8.2} fraction", "load 🏋️", load);
    csv_line.push_str(&format!("{load:.4},"));
    info!("{:<10}\t{:8.2}", "-sent", load_sent);
    csv_line.push_str(&format!("{load_sent:.4},"));
    info!("{:<10}\t{:8.2}", "-recv", load_recv);
    csv_line.push_str(&format!("{load_recv:.4},"));

    info!("");
    info!("{:<10}\t{:8.2} fraction", "delay ⏱️", delay);
    csv_line.push_str(&format!("{delay:.4},"));
    info!("{:<10}\t{:8.2} seconds", "-base", time_base);
    csv_line.push_str(&format!("{time_base:.4},"));
    info!("{:<10}\t{:8.2} seconds", "-defended", time_defended);
    csv_line.push_str(&format!("{time_defended:.4},"));

    println!();
}

#[allow(clippy::too_many_arguments)]
fn run_attacks(
    cfg: &EvalConfig,
    input: String,
    max: usize,
    n_classes: usize,
    n_samples: usize,
    fold: usize,
    csv_line: &mut String,
    seed: Option<u64>,
) {
    let epochs = cfg.epochs.unwrap_or(30);
    let patience = cfg.patience.unwrap_or(10);
    if let Some(rf) = &cfg.rf {
        info!(
            "{} Running RF {} with {} epochs and patience {} on fold {}...",
            Emoji("🧠", ""),
            rf,
            epochs,
            patience,
            fold
        );
        let accuracy = run_attack_script(
            &input,
            rf,
            "",
            n_classes,
            n_samples,
            epochs,
            patience,
            fold,
            &cfg.rf_model,
            seed,
        );
        info!("{} done, accuracy {:.2}", Emoji("🧠", ""), accuracy);
        csv_line.push_str(&format!("{accuracy:.4},"));
    } else {
        csv_line.push_str(&format!("{:.4},", -1.0));
    }

    if let Some(df) = &cfg.df {
        info!(
            "{} Running DF {} with {} epochs and patience {} on fold {}...",
            Emoji("🧠", ""),
            df,
            epochs,
            patience,
            fold
        );
        let flags = if max <= 5000 { "" } else { "-l" };
        let accuracy = run_attack_script(
            &input,
            df,
            flags,
            n_classes,
            n_samples,
            epochs,
            patience,
            fold,
            &cfg.df_model,
            seed,
        );
        info!("{} done, accuracy {:.2}", Emoji("🧠", ""), accuracy);
        csv_line.push_str(&format!("{accuracy:.4}"));
    } else {
        csv_line.push_str(&format!("{:.4}", -1.0));
    }
}

#[derive(Debug)]
struct TraceStats {
    sent_normal: i32,
    sent_padding: i32,
    recv_normal: i32,
    recv_padding: i32,
    duration_to_last_normal: Duration,
}

impl TraceStats {
    pub fn new() -> Self {
        TraceStats {
            sent_normal: 0,
            sent_padding: 0,
            recv_normal: 0,
            recv_padding: 0,
            duration_to_last_normal: Duration::from_secs(0),
        }
    }
}

fn get_trace_stats(
    trace: &str,
    max_events: usize,
    fname: &str,
    duration_at_n: usize,
) -> (TraceStats, Option<Duration>) {
    let mut stats = TraceStats::new();
    let mut requested_duration = None;

    // find the last normal packet
    let mut n = 0;
    let mut last_normal_packet = Duration::default();
    for line in trace.lines() {
        let line = line.trim();
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let duration_from_start = Duration::from_nanos(
                parts[0]
                    .trim()
                    .parse::<u64>()
                    .unwrap_or_else(|_| panic!("failed to parse timestamp in {fname}")),
            );

            match parts[1] {
                "sn" | "s" | "rn" | "r" => {
                    n += 1;
                    last_normal_packet = duration_from_start;
                    if n == duration_at_n {
                        requested_duration = Some(duration_from_start);
                    }
                }
                _ => {}
            };
        }
    }
    stats.duration_to_last_normal = last_normal_packet;
    if duration_at_n == 0 {
        requested_duration = Some(last_normal_packet);
    }

    for line in trace.lines().take(if max_events == 0 {
        usize::MAX
    } else {
        max_events
    }) {
        let line = line.trim();
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let duration_from_start = Duration::from_nanos(parts[0].trim().parse::<u64>().unwrap());

            // break if we've reached the last normal packet
            if duration_from_start > last_normal_packet {
                break;
            }

            match parts[1] {
                "sn" | "s" => {
                    stats.sent_normal += 1;
                }
                "sp" => {
                    stats.sent_padding += 1;
                }
                "rn" | "r" => {
                    stats.recv_normal += 1;
                }
                "rp" => {
                    stats.recv_padding += 1;
                }
                _ => {}
            };
        }
    }

    (stats, requested_duration)
}

#[allow(clippy::too_many_arguments)]
pub fn run_attack_script(
    dataset: &str,
    dl_script: &str,
    flags: &str,
    n_classes: usize,
    n_samples: usize,
    epochs: usize,
    patience: usize,
    fold: usize,
    model: &Option<String>,
    seed: Option<u64>,
) -> f64 {
    let mut output = Command::new(dl_script);
    output.arg("-d");
    output.arg(dataset);
    output.arg("-c");
    output.arg(n_classes.to_string()); // convert n_classes to string
    output.arg("-s");
    output.arg(n_samples.to_string()); // convert n_samples to string
    output.arg("--epochs");
    output.arg(epochs.to_string());
    output.arg("--patience");
    output.arg(patience.to_string());
    output.arg("-f");
    output.arg(fold.to_string());
    if let Some(model) = model {
        output.arg("--lm");
        output.arg(model);
    }
    if let Some(seed) = seed {
        output.arg("--seed");
        output.arg(seed.to_string());
    }
    output.arg("--train");
    if !flags.is_empty() {
        for flag in flags.split(' ') {
            output.arg(flag);
        }
    }
    debug!("running {output:?}");
    let result = output.output().expect("failed to execute process");
    let accuracy_str = String::from_utf8(result.stdout).unwrap();

    if !result.status.success() {
        warn!(
            "attack script failed with error: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        panic!("attack script failed");
    }

    // last line is the accuracy
    accuracy_str
        .split_whitespace()
        .last()
        .unwrap()
        .parse::<f64>()
        .unwrap()
}
