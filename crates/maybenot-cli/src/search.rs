use std::{
    fs::metadata,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use anyhow::{bail, Result};
use indicatif::ProgressBar;
use log::info;
use maybenot_gen::defense::Defense;

use rand::{thread_rng, Rng, RngCore};
use rand_seeder::Seeder;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    config::Config,
    derive::{derive_defense, DeriveConfig, DEFAULT_MAX_ATTEMPTS},
    get_progress_style, save_defenses,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchConfig {
    /// the number of defenses to search for
    pub n: usize,
    /// the seed for deterministic search
    pub seed: Option<u64>,
    /// the maximum duration for the search
    pub max_duration_sec: Option<usize>,
}

pub fn do_search(
    description: String,
    cfg: &Config,
    output: String,
    n: Option<usize>,
    max_duration_sec: Option<usize>,
    seed: Option<u64>,
) -> Result<()> {
    if cfg.search.is_none() {
        bail!("search config is missing");
    }
    if cfg.derive.is_none() {
        bail!("derive config is missing, required by search");
    }
    let search = cfg.clone().search.unwrap();
    let derive = cfg.clone().derive.unwrap();

    info!("derive config: {derive:#?}");

    if metadata(&output).is_ok() {
        bail!("output '{}' already exists", output);
    }
    if derive.fixed_client.is_some() && derive.fixed_server.is_some() {
        bail!("cannot fix both client and server");
    }
    let max_attempts = derive.max_attempts.unwrap_or(DEFAULT_MAX_ATTEMPTS);
    if max_attempts == 0 {
        bail!("max_attempts must be at least 1");
    }

    let fixed_client = derive.fixed_client.clone().unwrap_or_default();
    let fixed_server = derive.fixed_server.clone().unwrap_or_default();
    if !fixed_client.is_empty() && !fixed_server.is_empty() {
        bail!("cannot fix both client and server");
    }
    if !fixed_client.is_empty() {
        info!("fixing client to {fixed_client:?}");
    }
    if !fixed_server.is_empty() {
        info!("fixing server to {fixed_server:?}");
    }

    let n = n.unwrap_or(search.n);
    if n == 0 {
        bail!("n must be at least 1");
    }

    let max_duration_sec = max_duration_sec.unwrap_or(search.max_duration_sec.unwrap_or(0));
    let max_duration = if max_duration_sec > 0 {
        Some(Duration::from_secs(max_duration_sec as u64))
    } else {
        None
    };

    let mut defenses = vec![];
    // if a seed is provided either as an argument or in the search config
    // (priority to argument), use it to deterministically generate defenses,
    // otherwise use the system RNG
    if let Some(seed) = seed.or(search.seed) {
        info!(
            "deterministically searching for up to {n} defenses with {max_attempts} max attempts and seed {seed} for xoshiro256**..."
        );
        // we use the provided seed and the number of defenses to seed the
        // random number generator, because otherwise increasing or decreasing
        // the number of defenses would not change the defenses (unlike most
        // configuration values for deriving defenses)
        let mut rng: Xoshiro256StarStar = Seeder::from(format!("{seed}-{n}")).make_rng();
        for mut defense in gen_random_defenses_deterministic(n, max_duration, &derive, &mut rng) {
            defense.note = Some(format!("search seed {}, {}", seed, defense.note.unwrap()));
            defenses.push(defense);
        }
    } else {
        info!(
            "securely searching for up to {n} defenses with {max_attempts} max attempts using the system RNG..."
        );
        for mut defense in gen_random_defenses_secure(n, max_duration, &derive) {
            defense.note = Some(format!("secure search, {}", defense.note.unwrap()));
            defenses.push(defense);
        }
    }

    info!("done, found {}/{} defenses, saving...", defenses.len(), n);
    save_defenses(description, &defenses, &output)
}

/// Generate random defenses with a deterministic random number generator. This
/// is slower than `gen_random_defenses_secure` but can be useful for debugging,
/// testing, and reproducibility. Might be insecure, depending on threat model
/// and how the defenses are intended to be used (assumed public or secret?).
fn gen_random_defenses_deterministic<R: RngCore>(
    n: usize,
    max_duration: Option<Duration>,
    cfg: &DeriveConfig,
    rng: &mut R,
) -> Vec<Defense> {
    let base_seed: u64 = rng.gen();

    let bar = ProgressBar::new(n as u64);
    bar.set_style(get_progress_style());
    let starting_time = Instant::now();

    let mut defenses: Vec<Option<Defense>> = vec![];
    (0..n)
        .into_par_iter()
        .map(|defense_n| {
            let mut round = 0;
            loop {
                let seed = format!("{base_seed}-{defense_n}-{round}");
                let def = derive_defense(&seed, cfg).unwrap();
                if def.is_some() {
                    bar.inc(1);
                    return def;
                }
                if let Some(max_duration) = max_duration {
                    if starting_time.elapsed() >= max_duration {
                        return None;
                    }
                }
                round += 1;
            }
        })
        .collect_into_vec(&mut defenses);

    bar.finish_and_clear();
    // filter out any None values
    let mut defenses: Vec<Defense> = defenses.into_iter().flatten().collect();

    // sort defenses by id to deterministically order them
    defenses.sort_by(|a, b| a.id().cmp(b.id()));

    defenses
}

/// Generate random defenses with a secure random number generator. The faster
/// and preferred way to generate random defenses.
fn gen_random_defenses_secure(
    n: usize,
    max_duration: Option<Duration>,
    cfg: &DeriveConfig,
) -> Vec<Defense> {
    let results = Arc::new(Mutex::new(Vec::new()));
    let cancel = Arc::new(AtomicBool::new(false));

    let bar = ProgressBar::new(n as u64);
    bar.set_style(get_progress_style());
    let starting_time = Instant::now();

    (0..num_cpus::get()).into_par_iter().for_each(|_| {
        let mut rng = thread_rng();

        while !cancel.load(Ordering::SeqCst) {
            let def;
            loop {
                let defense_seed: u64 = rng.gen();
                if let Ok(d) = derive_defense(&format!("{defense_seed}"), cfg) {
                    def = d;
                    break;
                };

                if let Some(max_duration) = max_duration {
                    if starting_time.elapsed() >= max_duration {
                        cancel.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }

            let mut r = results.lock().unwrap();
            if r.len() < n {
                r.push(def);
            }
            bar.inc(1);

            if r.len() >= n {
                cancel.store(true, Ordering::SeqCst);
                break;
            }
        }
    });

    let defenses = results.lock().unwrap();
    defenses.iter().filter_map(|d| d.clone()).collect()
}
