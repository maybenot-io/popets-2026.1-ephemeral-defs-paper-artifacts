use std::time::Duration;

use anyhow::{bail, Result};
use log::{error, info, warn};
use maybenot::Machine;
use maybenot_gen::constraints::Range;
use maybenot_gen::{
    constraints::Constraints, defense::Defense, environment::Environment, random::random_machine,
};
use maybenot_machines::get_machine;
use maybenot_machines::StaticMachine;
use maybenot_simulator::network::Network;
use rand::Rng;
use rand::RngCore;
use rand_seeder::Seeder;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};

use crate::config::{Config, EnvParams, MachineParams};

/// The configuration for deriving a defense based on a seed.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DeriveConfig {
    /// the number of machines for the defense, per side
    pub num_machines: Range,
    /// the maximum number of attempts to derive a defense (default: 1024)
    pub max_attempts: Option<usize>,
    /// optionally hardcoded machines at the client, counts towards the number
    /// of client machines sampled (e.g., if we sample 3 machines and we have 1
    /// hardcoded machine, we sample 2 more machines)
    pub fixed_client: Option<Vec<StaticMachine>>,
    /// optionally hardcoded machines at the server, same restrictions as
    /// fixed_client apply
    pub fixed_server: Option<Vec<StaticMachine>>,
    /// parameters for the machines to be derived
    pub machine: MachineParams,
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
}

/// The default maximum number of attempts to derive a defense.
pub const DEFAULT_MAX_ATTEMPTS: usize = 1024;

pub fn do_derive(cfg: &Config, seed: String, verbose: bool) -> Result<()> {
    if cfg.derive.is_none() {
        bail!("derive config is missing");
    }
    if !seed.is_ascii() {
        bail!("seed must be an ascii string");
    }

    let cfg = cfg.derive.as_ref().unwrap();
    let max_attempts = cfg.max_attempts.unwrap_or(DEFAULT_MAX_ATTEMPTS);
    if max_attempts == 0 {
        bail!("max_attempts must be at least 1");
    }
    if cfg.env.traces.is_empty() {
        bail!("no traces provided in the environment");
    }

    info!("deriving defense from seed {seed} with at most {max_attempts} attempts...");

    match derive_defense(&seed, cfg)? {
        Some(defense) => {
            info!("done, defense:");
            print!("{defense}");
            if verbose {
                info!("client machine(s):");
                for m in &defense.client {
                    println!("{m}");
                }
                info!("server machine(s):");
                for m in &defense.server {
                    println!("{m}");
                }
            }
        }
        None => error!("no defense for the provided configuration and seed"),
    };

    Ok(())
}

/// Derive a defense from a seed and configuration. Note that the seed here is a
/// string that is later hashed before seeding the RNG. The input space is
/// bigger than the u64 seeds used for reproducibility in other parts of the
/// codebase.
pub fn derive_defense(seed: &str, cfg: &DeriveConfig) -> Result<Option<Defense>> {
    if cfg.fixed_client.is_some() && cfg.fixed_server.is_some() {
        bail!("cannot fix both client and server");
    }
    if !seed.is_ascii() {
        bail!("seed must be an ascii string");
    }

    let mut rng: Xoshiro256StarStar = Seeder::from(seed).make_rng();

    let env = Environment::new(
        &cfg.env.traces,
        cfg.env.num_traces.sample_usize(&mut rng),
        cfg.env.sim_steps.sample_usize(&mut rng),
        Network::new(
            Duration::from_millis((cfg.env.rtt_in_ms.sample_f64(&mut rng) / 2.0) as u64),
            cfg.env
                .packets_per_sec
                .map(|pps| pps.sample_usize(&mut rng)),
        ),
        cfg.env.integration.clone(),
        // the constrains serve as an upper bound for the padding and blocking
        // of the defenses: we can enforce this in the framework
        load2frac(cfg.constraints.client_load.1, cfg.no_implied_limits),
        load2frac(cfg.constraints.server_load.1, cfg.no_implied_limits),
        load2frac(cfg.constraints.delay.1, cfg.no_implied_limits),
        load2frac(cfg.constraints.delay.1, cfg.no_implied_limits),
        &mut rng,
    )?;

    let max_attempts = cfg.max_attempts.unwrap_or(DEFAULT_MAX_ATTEMPTS);

    // attempt to derive the defense
    match find_constrained_defense(cfg, &env, max_attempts, &mut rng)? {
        Some((mut defense, attempts)) => {
            defense.note = Some(format!(
                "seed {}, v{}, {} attempts",
                seed,
                env!("CARGO_PKG_VERSION"),
                attempts
            ));

            Ok(Some(defense))
        }
        None => Ok(None),
    }
}

pub fn load2frac(load: f64, no_implied_limits: Option<bool>) -> f64 {
    // if no_implied_limits is set and true, return 0.0
    if no_implied_limits.unwrap_or(false) {
        return 0.0;
    }

    // The load is expressed as #defended/#undefended packets, for the two
    // complete traces. The fraction in the framework is the fraction of
    // padding/normal packets. For example, a load of 1.0 means a fraction of
    // 0.5, and a load of 2.0 means a fraction of 0.6666666666666666.
    // Converting:
    load / (load + 1.0)
}

fn find_constrained_defense<R: RngCore>(
    cfg: &DeriveConfig,
    env: &Environment,
    max_attempts: usize,
    rng: &mut R,
) -> Result<Option<(Defense, usize)>> {
    let mut attempts = 0;

    loop {
        attempts += 1;
        if attempts > max_attempts {
            // we tried enough times, no defense found
            return Ok(None);
        }

        let fixed_client: Vec<Machine> = match &cfg.fixed_client {
            Some(s) => get_machine(s, rng),
            None => vec![],
        };
        let fixed_server: Vec<Machine> = match &cfg.fixed_server {
            Some(s) => get_machine(s, rng),
            None => vec![],
        };

        let mut client = fixed_client.to_vec();
        let n = cfg
            .num_machines
            .sample_usize(rng)
            .saturating_sub(fixed_client.len());
        for _ in 0..n {
            client.push(random_machine(
                cfg.machine.num_states.sample_usize(rng),
                cfg.machine
                    .allow_blocking_client
                    .unwrap_or(rng.gen_bool(0.5)),
                cfg.machine.allow_expressive.unwrap_or(rng.gen_bool(0.5)),
                cfg.machine.allow_fixed_budget.unwrap_or(rng.gen_bool(0.5)),
                cfg.machine.allow_frac_limits.unwrap_or(rng.gen_bool(0.5)),
                false,
                cfg.machine.duration_ref_point,
                cfg.machine.count_ref_point,
                cfg.machine.min_action_timeout,
                rng,
            ));
        }

        let mut server = fixed_server.to_vec();
        let n = cfg
            .num_machines
            .sample_usize(rng)
            .saturating_sub(fixed_server.len());
        for _ in 0..n {
            server.push(random_machine(
                cfg.machine.num_states.sample_usize(rng),
                cfg.machine
                    .allow_blocking_server
                    .unwrap_or(rng.gen_bool(0.5)),
                cfg.machine.allow_expressive.unwrap_or(rng.gen_bool(0.5)),
                cfg.machine.allow_fixed_budget.unwrap_or(rng.gen_bool(0.5)),
                cfg.machine.allow_frac_limits.unwrap_or(rng.gen_bool(0.5)),
                false,
                cfg.machine.duration_ref_point,
                cfg.machine.count_ref_point,
                cfg.machine.min_action_timeout,
                rng,
            ));
        }

        match cfg.constraints.check(&client, &server, env, rng.next_u64()) {
            Ok(_) => {
                return Ok(Some((Defense::new(client, server), attempts)));
            }
            Err(e) => {
                if cfg.debug.unwrap_or(false) {
                    warn!("Constraint check failed: {e}");
                }
            }
        }
    }
}
