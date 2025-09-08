use crate::dutils::{make_dataset_map, read_dataset};
use crate::load_trace_to_string;
use log::{info, warn};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::sync::Mutex;
use std::{path::Path, time::Duration};

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
// pub fn get_trace_content(path: &String) -> String {
//     std::fs::read_to_string(path).unwrap()
// }

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
        if parts.len() == 3 {
            let duration_from_start = Duration::from_nanos(
                parts[0]
                    .trim()
                    .parse::<u64>()
                    .unwrap_or_else(|_| panic!("failed to parse timestamp in {}", fname)),
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

pub fn compute_overheads(
    base_path: &Path,
    defended_path: &Path,
    max: usize,
    real_world: bool,
) -> HashMap<String, f64> {
    let base_dataset = read_dataset(base_path);
    let base_map: HashMap<String, String> = make_dataset_map(&base_dataset);

    let defended_dataset = read_dataset(defended_path);

    let capacity = base_map.len();

    let stats = Mutex::new(HashMap::new());
    {
        let mut stats_map = stats.lock().unwrap();
        for key in OVERHEAD_KEYS.iter() {
            stats_map.insert(*key, Vec::with_capacity(capacity));
        }
    }

    info!("computing overheads (max trace length {})...", max);
    defended_dataset
        .par_iter()
        .for_each(|(class, fname, trace)| {
            // either base dataset has the fname, or we skip
            let base_key = format!("{}+{}", class, fname);
            let base_fname = match base_map.get(&base_key) {
                Some(base_fname) => base_fname,
                None => {
                    return;
                }
            };
            let trace = &load_trace_to_string(trace).unwrap();

            let (defended, _) = get_trace_stats(trace, max, fname, 0);
            if defended.sent_normal == 0 && defended.recv_normal == 0 {
                warn!("no normal traffic in {}, skipping", fname);
                return;
            }
            let base_duration_at_n = if real_world {
                0
            } else {
                (defended.sent_normal + defended.recv_normal) as usize
            };
            let base_trace = &load_trace_to_string(base_fname).unwrap();
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
                warn!("defended stats {:?}", defended);
                warn!("base stats {:?}", base);
                panic!("missing base duration for class {} fname {}, implies extra normal traffic after defense simulation", class, fname);
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

    let vals: Vec<f64> = vec![
        base,
        base_sent,
        base_recv,
        defended,
        defended_sent,
        defended_recv,
        missing,
        missing_sent,
        missing_recv,
        load,
        load_sent,
        load_recv,
        delay,
        time_base,
        time_defended,
    ];

    let dict: HashMap<String, f64> = OVERHEAD_KEYS
        .into_iter()
        .zip(vals.into_iter())
        .map(|(k, v)| (k.to_string(), v))
        .collect();

    dict
}
