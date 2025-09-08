use std::{cmp, fs::metadata, time::Duration};

use anyhow::{bail, Result};
use indicatif::ProgressBar;
use log::{info, warn};
use maybenot::Machine;
use maybenot_gen::{constraints::Constraints, defense::Defense, environment::Environment};
use maybenot_simulator::network::Network;
use num_bigint::BigUint;
use num_integer::binomial;
use num_traits::Zero;
use rand::{seq::SliceRandom, Rng, RngCore};
use rand_seeder::Seeder;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    config::{Config, EnvParams},
    derive::load2frac,
    get_progress_style, load_defenses, save_defenses,
};

/// The configuration for creating defenses by combining machines
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ComboConfig {
    /// the number of defenses to create
    pub n: Option<usize>,
    /// the maximum number of machines per defense
    pub height: Option<usize>,
    /// ignore constraints when combining machines as new defenses
    pub ignore_constraints: Option<bool>,
    /// the environment that defines the search space, consisting of
    /// representative network traces, network conditions between client and
    /// server, and potential integration peculiarities at the client and server
    pub env: EnvParams,
    /// the constraints placed on the defense to be considered valid
    pub constraints: Constraints,
    /// do not set implied simulation limits from constraints when deriving
    pub no_implied_limits: Option<bool>,
    /// print why constraints failed
    pub debug: Option<bool>,
    /// the maximum number of machines to use from defenses
    pub max_machines: Option<usize>,
    /// the maximum number of attempts to find a constrained defense before
    /// sampling another environment (default: 100)
    pub max_attempts: Option<usize>,
}

pub fn do_combo(
    input: Vec<String>,
    output: String,
    cfg: &Config,
    n: Option<usize>,
    height: Option<usize>,
    seed: Option<u64>,
) -> Result<()> {
    if cfg.combo.is_none() {
        bail!("combo config is missing");
    }
    let cfg = cfg.combo.as_ref().unwrap();
    if cfg.env.traces.is_empty() {
        bail!("no traces provided in the environment");
    }
    info!("combo config: {cfg:#?}");
    if input.is_empty() {
        bail!("no input files provided");
    }
    if metadata(&output).is_ok() {
        bail!("output '{}' already exists", output);
    }

    let n = n.or(cfg.n);
    if n.is_none() || n.unwrap() == 0 {
        bail!("number of defenses to create must be at least 1");
    }
    let n = n.unwrap();

    let height = height.or(cfg.height);
    if height.is_none() || height.unwrap() == 0 {
        bail!("height of defenses must be at least 1");
    }
    let height = height.unwrap();

    let seed = seed.unwrap_or(0);
    info!("deterministic, using seed {seed}");
    let mut rng: Xoshiro256StarStar = Seeder::from(seed).make_rng();

    let mut clients = Vec::new();
    let mut servers = Vec::new();
    for i in input.clone() {
        info!("loading defenses from {i}...");
        let read = load_defenses(&i)?;
        info!("loaded {} defenses from '{}'", read.defenses.len(), i);
        for d in read.defenses {
            clients.extend(d.client);
            servers.extend(d.server);
        }
    }
    info!(
        "in total {} client machines and {} server machines",
        clients.len(),
        servers.len()
    );
    if let Some(max_machines) = cfg.max_machines {
        // prune
        clients.truncate(max_machines);
        servers.truncate(max_machines);
        info!(
            "pruned to {} client machines and {} server machines",
            clients.len(),
            servers.len()
        );
    }

    let possible =
        count_stacked_combinations_deck_combinations(clients.len(), servers.len(), height);
    info!(
        "possible combinations with height {}: {}",
        height,
        big_uint_to_scientific(&possible, 3)
    );

    // randomize the order of clients and servers
    clients.shuffle(&mut rng);
    servers.shuffle(&mut rng);

    info!("creating {n} defenses with height {height}...");
    let bar = ProgressBar::new(n as u64);
    bar.set_style(get_progress_style());

    let mut defenses = vec![];
    let mut total_attempts: Vec<usize> = Vec::new();
    let mut remaining = n;

    loop {
        let base_seed: u64 = rng.gen();
        let collected = (0..remaining)
            .into_par_iter()
            .map(|n| {
                let mut round = 0;
                loop {
                    // create a new RNG for each defense for reproducibility
                    let mut rng: Xoshiro256StarStar =
                        Seeder::from(format!("{base_seed}-{n}-{round}")).make_rng();

                    if cfg.ignore_constraints.unwrap_or(false) {
                        // easy mode: ignore constraints
                        let n_client = rng.gen_range(1..=height);
                        let n_server = rng.gen_range(1..=height);

                        let client: Vec<Machine> = clients
                            .choose_multiple(&mut rng, n_client)
                            .cloned()
                            .collect();
                        let server: Vec<Machine> = servers
                            .choose_multiple(&mut rng, n_server)
                            .cloned()
                            .collect();
                        bar.inc(1);
                        return (Defense::new(client, server), 0);
                    }

                    let env = Environment::new(
                        &cfg.env.traces,
                        cfg.env.num_traces.sample_usize(&mut rng),
                        cfg.env.sim_steps.sample_usize(&mut rng),
                        Network::new(
                            Duration::from_millis(
                                (cfg.env.rtt_in_ms.sample_f64(&mut rng) / 2.0) as u64,
                            ),
                            cfg.env
                                .packets_per_sec
                                .map(|pps| pps.sample_usize(&mut rng)),
                        ),
                        cfg.env.integration.clone(),
                        // the constrains serve as an upper bound for the padding and
                        // blocking of the defenses: we can enforce this in the
                        // framework
                        load2frac(cfg.constraints.client_load.1, cfg.no_implied_limits),
                        load2frac(cfg.constraints.server_load.1, cfg.no_implied_limits),
                        load2frac(cfg.constraints.delay.1, cfg.no_implied_limits),
                        load2frac(cfg.constraints.delay.1, cfg.no_implied_limits),
                        &mut rng,
                    )
                    .unwrap();

                    let mut attempts = 0;

                    while attempts < cfg.max_attempts.unwrap_or(100) {
                        attempts += 1;
                        let n_client = rng.gen_range(1..=height);
                        let n_server = rng.gen_range(1..=height);

                        let client: Vec<Machine> = clients
                            .choose_multiple(&mut rng, n_client)
                            .cloned()
                            .collect();
                        let server: Vec<Machine> = servers
                            .choose_multiple(&mut rng, n_server)
                            .cloned()
                            .collect();

                        match cfg
                            .constraints
                            .check(&client, &server, &env, rng.next_u64())
                        {
                            Ok(_) => {
                                bar.inc(1);
                                return (Defense::new(client, server), attempts);
                            }
                            Err(e) => {
                                if cfg.debug.unwrap_or(false) {
                                    warn!("Constraint check failed: {e}");
                                }
                            }
                        }
                    }
                    round += 1;
                }
            })
            .collect::<Vec<_>>();

        total_attempts.extend(collected.iter().map(|&(_, a)| a));
        defenses.extend(collected.into_iter().map(|(d, _)| d));

        // sort vector by id to deterministically order them, then remove duplicates
        defenses.sort_by(|a, b| a.id().cmp(b.id()));
        defenses.dedup_by(|a, b| a.id() == b.id());
        remaining = n - defenses.len();
        if remaining == 0 {
            break;
        }
        info!("removed {remaining} duplicate defenses, creating new ones...");
        bar.set_position((n - remaining) as u64);
    }

    bar.finish_and_clear();

    // attempts statistics
    let mut attempts: Vec<f64> = total_attempts
        .into_iter()
        .map(|a: usize| a as f64)
        .collect();
    attempts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = attempts.iter().sum();
    let mean = sum / attempts.len() as f64;
    let median = attempts[attempts.len() / 2];
    let max = attempts[attempts.len() - 1];
    let min = attempts[0];
    let q1 = attempts[attempts.len() / 4];
    let q3 = attempts[(attempts.len() * 3) / 4];
    info!(
        "attempts statistics (constraints strict?): mean {mean:.2}, median {median}, min {min}, q1 {q1}, q3 {q3}, max {max}"
    );

    info!("done, saving...");
    save_defenses("combo".to_owned(), &defenses, &output)?;
    Ok(())
}

/// Computes the number of possible combinations when picking between 1 and M
/// items from each of two lists.
pub fn count_stacked_combinations_deck_combinations(
    a_size: usize,
    b_size: usize,
    m: usize,
) -> BigUint {
    // Sum of binomial coefficients for choosing 1 to M elements from A
    let s_a: BigUint = (1..=cmp::min(a_size, m))
        .map(|k| binomial(BigUint::from(a_size), BigUint::from(k)))
        .sum();

    // Sum of binomial coefficients for choosing 1 to M elements from B
    let s_b: BigUint = (1..=cmp::min(b_size, m))
        .map(|k| binomial(BigUint::from(b_size), BigUint::from(k)))
        .sum();

    // Total combinations is the Cartesian product of S_A and S_B
    s_a * s_b
}

// Convert a BigUint to a string in scientific notation.
/// `precision` is the number of digits to show in the mantissa (excluding the decimal point).
pub fn big_uint_to_scientific(n: &BigUint, precision: usize) -> String {
    if n.is_zero() {
        return "0e+0".to_string();
    }

    // Get the number as a string
    let s = n.to_string();
    let len = s.len();

    // The exponent is the position of the first digit (0-indexed) in a number where only one digit is left of the decimal point.
    let exponent = len - 1;

    // Prepare the mantissa.
    // The first digit is always shown. We then include the next (precision - 1) digits (if available) after a decimal point.
    let first_digit = &s[0..1];
    let remainder = if len > 1 {
        // Ensure we don't exceed the string length
        &s[1..std::cmp::min(len, precision)]
    } else {
        ""
    };

    // Build the formatted mantissa.
    let mantissa = if remainder.is_empty() {
        first_digit.to_string()
    } else {
        format!("{first_digit}.{remainder}")
    };

    format!("{mantissa}e+{exponent}")
}
