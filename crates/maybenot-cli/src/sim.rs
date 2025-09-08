use anyhow::{bail, Result};
use rand_seeder::Seeder;
use rand_xoshiro::Xoshiro256StarStar;
use std::{
    fs::metadata,
    path::Path,
    time::{Duration, Instant},
};

use indicatif::ParallelProgressIterator;
use log::{info, warn};
use maybenot::TriggerEvent;
use maybenot_gen::{
    constraints::Range,
    dealer::{Dealer, DealerFixed, Limits},
    defense::Defense,
    environment::{get_example_client, get_example_server, integration_from_file, IntegrationType},
};
use maybenot_simulator::{network::Network, parse_trace_advanced, sim_advanced, SimulatorArgs};
use rand::{seq::SliceRandom, Rng, RngCore, SeedableRng};
use rayon::{
    iter::IndexedParallelIterator,
    prelude::{IntoParallelRefIterator, ParallelIterator},
};
use serde::{Deserialize, Serialize};

use crate::{config::Config, get_progress_style, get_trace_content, load_defenses, read_dataset};

// shared config struct for simulator
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct SimulatorConfig {
    /// client limits, if not set, uses no limits
    pub client: Option<Limits>,
    /// server limits, if not set, uses no limits
    pub server: Option<Limits>,
    /// sampled rtt between client and server
    pub rtt_in_ms: Range,
    /// sampled bottleneck between client and server
    pub packets_per_sec: Option<Range>,
    /// trace length to simulate
    pub trace_length: usize,
    /// extra events in the simulator, set higher for more complex defenses
    pub events_multiplier: usize,
    /// integration type
    pub integration: Option<IntegrationType>,
    /// stop after all normal packets are processed, default is false
    pub stop_after_all_normal_packets_processed: Option<bool>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct SimConfig {
    /// probability of not using a defense
    pub undefended_prob: Option<f64>,
    /// base dataset directory
    pub base_dataset: String,
    /// maximum number of samples to simulate per subpage
    pub max_samples: usize,
    /// simulate each trace multiple times
    pub augmentation: Option<usize>,
    /// only take a number of defenses from the input, after filtering, sorted
    /// by defense identifier (for reproducibility, basically deterministically
    /// random selection since the identifier is derived from the serialized
    /// representation of all its machines)
    pub take_defenses: Option<usize>,
    /// allow duplicate defenses in the input
    pub allow_duplicates: Option<bool>,
    /// defense simulator configuration
    pub simulator: SimulatorConfig,
    /// simulate with tunable defense limits using a list of factors [0,1] to
    /// multiply limits set in the simulator config
    pub tunable_defense_limits: Option<Vec<f64>>,
    /// seed for deterministic simulation
    pub seed: Option<u64>,
}

pub fn do_sim(
    cfg: &Config,
    input: Vec<String>,
    output: String,
    take_defenses: Option<usize>,
    seed: Option<u64>,
) -> Result<()> {
    if cfg.sim.is_none() {
        bail!("sim config is missing");
    }
    let cfg = cfg.sim.as_ref().unwrap();
    info!("using {cfg:#?}");

    if let Some(limits) = &cfg.tunable_defense_limits {
        if limits.is_empty() {
            bail!("tunable defense limits is empty");
        }
        for l in limits.iter() {
            if *l < 0.0 || *l > 1.0 {
                bail!("tunable defense limit must be in [0,1], found {}", l);
            }
        }
    }

    // safe to always use Xoshiro256StarStar, since there's nothing adversarial
    // to learn here from simulation output
    let mut rng = match &seed.or(cfg.seed) {
        Some(seed) => {
            info!("deterministic, using seed {seed}");
            Seeder::from(seed).make_rng()
        }
        None => Xoshiro256StarStar::from_entropy(),
    };

    let mut defenses = Vec::new();
    for i in input {
        info!("loading defenses from {i}...");
        let mut read = load_defenses(&i)?;
        // update the id of each defense
        for d in read.defenses.iter_mut() {
            d.update_id();
        }
        info!("read {} defenses from {}", read.defenses.len(), i);
        defenses.extend(read.defenses);
    }

    if !cfg.allow_duplicates.unwrap_or(false) {
        // remove duplicates based on defense id
        let num_defenses = defenses.len();
        defenses.sort_by(|a, b| a.id().cmp(b.id()));
        defenses.dedup_by(|a, b| a.id() == b.id());
        if num_defenses != defenses.len() {
            info!(
                "removed {} duplicate defenses",
                num_defenses - defenses.len()
            );
        }
        info!("read {} unique defenses in total", defenses.len());
    } else {
        info!("read {} defenses in total", defenses.len());
    }

    if let Some(n) = take_defenses.or(cfg.take_defenses) {
        // sort by defense id, then deterministically shuffle
        defenses.sort_by(|a, b| a.id().cmp(b.id()));
        defenses.shuffle(&mut rng);
        defenses.truncate(n);
        info!("took {n} defenses");
    }

    // make sure that we have at least one defense
    if defenses.is_empty() {
        bail!("no defenses left after filtering/take");
    }

    if let Some(prob) = cfg.undefended_prob {
        if !(0.0..1.0).contains(&prob) {
            bail!("undefended probability must be in [0,1], found {}", prob);
        }
        info!("undefended probability {prob}");
    }

    do_sim_def(cfg, &mut defenses, output, &mut rng)
}

fn do_sim_def<R: RngCore>(
    cfg: &SimConfig,
    defenses: &mut [Defense],
    output: String,
    rng: &mut R,
) -> Result<()> {
    // make sure that output does not exist, then create the directory
    if metadata(&output).is_ok() {
        bail!("output {} already exists", output);
    }
    std::fs::create_dir(&output)?;

    info!("reading base dataset from {}", cfg.base_dataset);
    let base_dataset = std::path::Path::new(&cfg.base_dataset);
    if !base_dataset.exists() {
        bail!("base dataset {} does not exist", base_dataset.display());
    }
    if !base_dataset.is_dir() {
        bail!("base dataset {} is not a directory", base_dataset.display());
    }
    let dataset = read_dataset(base_dataset);
    info!("read {} traces", dataset.len());

    let mut dataset_samples = 0;
    for (_, fname, _) in &dataset {
        let parts = fname.split('-').collect::<Vec<_>>();
        // we assume that the filename ends with sample.log
        let sample: usize = parts
            .last()
            .unwrap()
            .split('.')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        if sample + 1 > dataset_samples {
            dataset_samples = sample + 1;
        }
    }

    // randomize defenses
    defenses.shuffle(rng);

    if let Some(aug) = cfg.augmentation {
        if aug == 0 {
            bail!("augmentation must be at least 1");
        }
        if aug > 1 {
            info!("augmenting dataset {aug} times");
        }
        // check for edge case: augmentation set and max_samples is below the
        // number of (non-augmented) samples per subpage
        if cfg.max_samples > 0 && cfg.max_samples < dataset_samples {
            bail!(
                "augmentation set, but max_samples < |samples| in dataset {}",
                dataset_samples
            );
        }
    }

    let enough_defenses = defenses.len() >= dataset.len() * cfg.augmentation.unwrap_or(1);
    if enough_defenses {
        info!("enough defenses to cover dataset");
    } else {
        info!("not enough defenses to cover dataset, will randomly choose");
    }

    if cfg.tunable_defense_limits.is_none() {
        info!("simulating...");
        sim_dataset(
            &dataset,
            dataset_samples,
            enough_defenses,
            cfg,
            defenses,
            0.0,
            &output,
            rng,
        )?;
        info!(
            "done, wrote {} traces to {}",
            dataset.len() * cfg.augmentation.unwrap_or(1),
            output
        );
    } else {
        let limits = cfg.tunable_defense_limits.as_ref().unwrap();
        for limit in limits.iter() {
            // output is a path: create a subdirectory for the limit
            let output = Path::new(&output).join(format!("limit-{limit}"));
            let output = output.to_str().unwrap();

            // simulate
            info!("simulating with limit {limit}...");
            sim_dataset(
                &dataset,
                dataset_samples,
                enough_defenses,
                cfg,
                defenses,
                *limit,
                output,
                rng,
            )?;
            info!(
                "done, wrote {} traces to {}",
                dataset.len() * cfg.augmentation.unwrap_or(1),
                output
            );
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sim_dataset<R: RngCore>(
    dataset: &[(usize, String, String)],
    dataset_samples: usize,
    enough_defenses: bool,
    cfg: &SimConfig,
    defenses: &[Defense],
    scale: f64,
    output: &str,
    rng: &mut R,
) -> Result<()> {
    let undefended_prob = cfg.undefended_prob.unwrap_or(0.0);

    let (c_integration, s_integration) = match cfg.simulator.integration {
        Some(IntegrationType::Example) => (get_example_client(), get_example_server()),
        Some(IntegrationType::File { ref src }) => {
            let (c, s) = integration_from_file(src)?;
            (Some(c), Some(s))
        }
        None => (None, None),
    };

    let augmentation = cfg.augmentation.unwrap_or(1);

    // create a dealer for client and server setups
    let mut dealer = DealerFixed::new(
        defenses.to_vec(),
        cfg.simulator.client.clone(),
        cfg.simulator.server.clone(),
        !enough_defenses,
        rng,
    )?;
    let setups = dealer.draw_n(dataset.len() * augmentation, scale, rng)?;

    // we need to know if the original dataset filenames are zero-padded or not
    // to generate correct filenames for the augmented traces
    let is_zero_padded = find_if_zero_padded(dataset);

    // shared seed for all traces ...
    let shared_seed: u64 = rng.gen();

    dataset
        .par_iter()
        .enumerate()
        .progress_with_style(get_progress_style())
        .for_each(|(index, (class, fname, trace_path))| {
            // ... combined with a per-trace seed using the trace class+fname,
            // which is unique for our dataset structure
            let mut rng: Xoshiro256StarStar = Seeder::from(format!(
                "shared seed {shared_seed}, class {class}, fname {fname}"
            ))
            .make_rng();

            // we assume that the filename ends with sample.log
            let parts = fname.split('-').collect::<Vec<_>>();
            let sample: usize = parts
                .last()
                .unwrap()
                .split('.')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            // prefix is everything before the sample number and extension
            let prefix = parts[..parts.len() - 1].join("-");
            // early exit?
            if cfg.max_samples > 0 && sample >= cfg.max_samples {
                return;
            }
            let base_trace = &get_trace_content(trace_path);

            for a in 0..augmentation {
                let mut setup = setups[index + dataset.len() * a].clone();
                if undefended_prob > 0.0 && rng.gen_bool(undefended_prob) {
                    setup.client.machines.clear();
                    setup.server.machines.clear();
                }

                // random delay and packets per second
                let network = Network::new(
                    Duration::from_millis(
                        (cfg.simulator.rtt_in_ms.sample_f64(&mut rng) / 2.0) as u64,
                    ),
                    cfg.simulator
                        .packets_per_sec
                        .map(|pps| pps.sample_usize(&mut rng)),
                );

                // parse content into a pq and sim
                let mut pq = parse_trace_advanced(
                    base_trace,
                    network,
                    c_integration.as_ref(),
                    s_integration.as_ref(),
                );
                let trace = sim_advanced(
                    &setup.client.machines,
                    &setup.server.machines,
                    &mut pq,
                    &SimulatorArgs {
                        network,
                        max_trace_length: cfg.simulator.trace_length,
                        max_sim_iterations: cfg.simulator.trace_length
                            * cfg.simulator.events_multiplier,
                        continue_after_all_normal_packets_processed: !cfg
                            .simulator
                            .stop_after_all_normal_packets_processed
                            .unwrap_or(false),
                        only_client_events: true,
                        only_network_activity: true,
                        max_padding_frac_client: setup.client.max_padding_frac,
                        max_blocking_frac_client: setup.client.max_blocking_frac,
                        max_padding_frac_server: setup.server.max_padding_frac,
                        max_blocking_frac_server: setup.server.max_blocking_frac,
                        insecure_rng_seed: Some(rng.next_u64()),
                        client_integration: c_integration.clone(),
                        server_integration: s_integration.clone(),
                    },
                );

                // in trace, filter out the events at the client
                if trace.is_empty() {
                    warn!("no client events in trace from {fname}, skipping");
                }
                let starting_time = if !trace.is_empty() {
                    trace[0].time
                } else {
                    Instant::now()
                };

                let mut s = String::with_capacity(cfg.simulator.trace_length * 20);
                let mut n: usize = 0;
                for t in trace {
                    if n > cfg.simulator.trace_length {
                        warn!("trace too long, truncating, broken sim args?");
                        break;
                    }

                    // timestamp, nanoseconds granularity (for consistency)
                    let ts = &format!("{}", t.time.duration_since(starting_time).as_nanos());

                    match t.event {
                        TriggerEvent::TunnelRecv => {
                            n += 1;
                            if t.contains_padding {
                                s.push_str(&format!("{ts},rp,514\n"));
                            } else {
                                s.push_str(&format!("{ts},rn,514\n"));
                            }
                        }
                        TriggerEvent::TunnelSent => {
                            n += 1;
                            if t.contains_padding {
                                s.push_str(&format!("{ts},sp,514\n"));
                            } else {
                                s.push_str(&format!("{ts},sn,514\n"));
                            }
                        }
                        _ => {}
                    };
                }

                // new fname, taking augmentation into account
                let new_fname = if !prefix.is_empty() {
                    if is_zero_padded {
                        format!("{}-{:04}.log", prefix, sample + (a * dataset_samples))
                    } else {
                        format!("{}-{}.log", prefix, sample + (a * dataset_samples))
                    }
                } else if is_zero_padded {
                    format!("{:04}.log", sample + (a * dataset_samples))
                } else {
                    format!("{}.log", sample + (a * dataset_samples))
                };

                // write to file
                let outdir = std::path::Path::new(&output).join(format!("{class}"));
                std::fs::create_dir_all(&outdir).unwrap();
                let outfile = outdir.join(new_fname);
                std::fs::write(outfile, s).unwrap();
            }
        });
    Ok(())
}

fn find_if_zero_padded(dataset: &[(usize, String, String)]) -> bool {
    let mut is_zero_padded = false;
    for (_, fname, _) in dataset {
        // HACK: find more suitable way
        if fname.ends_with("-01.log") || fname.ends_with("-001.log") || fname.ends_with("-0001.log")
        {
            is_zero_padded = true;
            break;
        }
    }

    is_zero_padded
}
