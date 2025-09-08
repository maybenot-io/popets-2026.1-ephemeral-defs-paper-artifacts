use std::{
    fs::{create_dir, metadata, read_to_string, remove_dir_all},
    path::Path,
};

use crate::{config::Config, eval::do_eval, load_defenses, search::do_search, sim::do_sim};
use anyhow::{bail, Result};
use log::info;
use maybenot_gen::{constraints::Range, environment::Traces, random::round_f64};
use rand::{seq::SliceRandom, Rng};
use rand_seeder::Seeder;
use rand_xoshiro::Xoshiro256StarStar;

pub fn do_tune_rng(
    cfg: &Config,
    probability: f64,
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
    if cfg.sim.is_none() {
        bail!("sim config is missing, required by search");
    }
    if cfg.eval.is_none() {
        bail!("eval config is missing, required by search");
    }
    if !(0.0..=1.0).contains(&probability) {
        bail!("probability must be in [0, 1]");
    }
    if metadata(&output).is_ok() {
        bail!("output {} already exists", output);
    }
    let n_check = n.unwrap_or(cfg.search.as_ref().unwrap().n);
    if n_check == 0 {
        bail!("n must be at least 1");
    }

    let cfg_is_cf = if cfg
        .derive
        .as_ref()
        .unwrap()
        .env
        .traces
        .contains(&Traces::TorCircuit)
    {
        info!("using circuit fingerprinting traces");
        true
    } else {
        info!("using web fingerprinting traces");
        false
    };

    let seed = seed.unwrap_or(0);
    let cfg_str = serde_json::to_string(cfg)
        .map_err(|e| anyhow::anyhow!("failed to serialize config for seeding: {}", e))?;
    let combined_seed = format!("{seed}{cfg_str}");
    let mut rng: Xoshiro256StarStar = Seeder::from(combined_seed).make_rng();
    info!("deterministic, using seed {seed} combined with the serialized configuration file");

    let csv = cfg.eval.as_ref().unwrap().csv.as_ref().unwrap();

    let original = cfg.clone();

    loop {
        info!(
            "🧂🧂 start of random tune loop, probability {probability}, checking for new config... 🧂🧂"
        );
        let mut cfg = original.clone();

        // TODO: wiggle sim as well?

        // make some change with some probability only if the results file is
        // here: if not, we run with unchanged config as a baseline
        let mut fname = "starting-config".to_string();
        if metadata(csv).is_ok() {
            let contents = read_to_string(csv)?;
            loop {
                // found a new config?
                if !contents.contains(&fname) {
                    break;
                }
                // wiggle the config
                cfg = original.clone();
                fname = if cfg_is_cf {
                    wiggle_cf_cfg(&mut cfg, probability, &mut rng)
                } else {
                    wiggle_wf_cfg(&mut cfg, probability, &mut rng)
                };

                // lazy csv escape, removing "," with good formatting
                fname = fname.replace(", ", "..");
                fname = fname.replace(",", "..");
            }
        };

        // create the output folder
        create_dir(&output)?;

        info!("🧂🧂 searching with def: {fname}");
        let def = Path::new(&output)
            .join(format!("{fname}.def"))
            .to_str()
            .unwrap()
            .to_owned();
        do_search(
            "rng tune".to_owned(),
            &cfg,
            def.clone(),
            n,
            max_duration_sec,
            None,
        )?;

        let loaded = load_defenses(&def)?;
        if loaded.defenses.is_empty() {
            info!("🧂🧂 no defenses found, removing tmp output and trying again");
            remove_dir_all(&output)?;
            continue;
        }

        let sim_folder = Path::new(&output)
            .join(format!("{}-{}", loaded.defenses.len(), fname))
            .to_str()
            .unwrap()
            .to_owned();

        // sim and eval
        info!("🧂🧂 sim for def: {fname}");
        do_sim(&cfg, vec![def.clone()], sim_folder.clone(), None, None)?;
        let sim = cfg.clone().sim.unwrap();
        info!("🧂🧂 eval for def: {fname}");
        if sim.tunable_defense_limits.is_none() {
            do_eval(&cfg, sim_folder.clone(), None)?;
        } else {
            let limits = sim.tunable_defense_limits.as_ref().unwrap();
            for limit in limits.iter() {
                let output = Path::new(&sim_folder).join(format!("limit-{limit}"));
                do_eval(&cfg, output.to_str().unwrap().to_owned(), None)?;
            }
        }
        info!("🧂🧂 done, removing tmp output");
        remove_dir_all(&output)?;
    }
}

fn wiggle_wf_cfg(cfg: &mut Config, prob: f64, rng: &mut Xoshiro256StarStar) -> String {
    // what we want to wiggle with some probability: attempts, states,
    // allow_frac_limits, duration ref point, count int ref point, min action
    // timeout, traces, num_traces, sim steps, client load, server load, delay,
    // client min normal packets, server min normal packets, include after last
    // normal

    // for each change we make, we build a new filename
    let mut fname = String::new();

    let derive = cfg.derive.as_mut().unwrap();
    let machine = &mut derive.machine;
    let env = &mut derive.env;
    let constraints = &mut derive.constraints;

    // attempts
    if rng.gen_bool(prob) {
        derive.max_attempts = Some(rng.gen_range(1..1024));
        fname.push_str(format!("att{}", derive.max_attempts.unwrap()).as_str());
    }

    // states
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=5);
        let min = rng.gen_range(1..=max);
        machine.num_states = Range(min as f64, max as f64);
        fname.push_str(format!("states{}", machine.num_states).as_str());
    }

    // allow_frac_limits
    if rng.gen_bool(prob) {
        machine.allow_frac_limits = Some(rng.gen_bool(0.5));
        fname.push_str(format!("fl{}", machine.allow_frac_limits.unwrap()).as_str());
    }

    // duration ref point
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=1_000_000);
        let min = rng.gen_range(1..=max);
        machine.duration_ref_point = Some(Range(min as f64, max as f64));
        fname.push_str(format!("drp{}", machine.duration_ref_point.unwrap()).as_str());
    }
    // count ref point
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=1_000);
        let min = rng.gen_range(1..=max);
        machine.count_ref_point = Some(Range(min as f64, max as f64));
        fname.push_str(format!("crp{}", machine.count_ref_point.unwrap()).as_str());
    }
    // min action timeout
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=1_000);
        let min = rng.gen_range(1..=max);
        machine.min_action_timeout = Some(Range(min as f64, max as f64));
        fname.push_str(format!("mat{}", machine.min_action_timeout.unwrap()).as_str());
    }

    // traces
    let mut updated_traces = false;
    if rng.gen_bool(prob) {
        // clone the original traces (it's a superset)
        let mut traces = env.traces.clone();
        traces.shuffle(rng);
        let num_traces = rng.gen_range(1..=traces.len());
        traces.truncate(num_traces);
        env.traces = traces;
        fname.push_str(format!("traces{:?}", env.traces).as_str());

        updated_traces = true;
    }

    // num traces
    if updated_traces || rng.gen_bool(prob) {
        // FIXME: magic constant 14, correct for now for WF but not for other
        // types of traces
        let m = env.traces.len() * 14;
        let max = rng.gen_range(1..=m);
        let min = rng.gen_range(1..=max);
        env.num_traces = Range(min as f64, max as f64);
        fname.push_str(format!("num{}", env.num_traces).as_str());
    }

    // sim steps
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=100_000);
        let min = rng.gen_range(1..=max);
        env.sim_steps = Range(min as f64, max as f64);
        fname.push_str(format!("steps{}", env.sim_steps).as_str());
    }

    // client load
    if rng.gen_bool(prob) {
        let max = rng.gen_range(0.0..=10.0);
        let min = rng.gen_range(0.0..=max);
        constraints.client_load = Range(round_f64(min), round_f64(max));
        fname.push_str(format!("cl{:?}", constraints.client_load).as_str());
    }
    // server load
    if rng.gen_bool(prob) {
        let max = rng.gen_range(0.0..=10.0);
        let min = rng.gen_range(0.0..=max);
        constraints.server_load = Range(round_f64(min), round_f64(max));
        fname.push_str(format!("sl{:?}", constraints.server_load).as_str());
    }
    // delay
    if constraints.delay.1 > 0.0 && rng.gen_bool(prob) {
        let max = rng.gen_range(0.0..=5.0);
        let min = rng.gen_range(0.0..=max);
        constraints.delay = Range(round_f64(min), round_f64(max));
        fname.push_str(format!("d{:?}", constraints.delay).as_str());
    }

    // client min normal packets
    if rng.gen_bool(prob) {
        constraints.client_min_normal_packets = Some(rng.gen_range(0..=80));
        fname.push_str(format!("cmin{}", constraints.client_min_normal_packets.unwrap()).as_str());
    }
    // server min normal packets
    if rng.gen_bool(prob) {
        constraints.server_min_normal_packets = Some(rng.gen_range(0..=200));
        fname.push_str(format!("smin{}", constraints.server_min_normal_packets.unwrap()).as_str());
    }
    // include after last normal
    if rng.gen_bool(prob) {
        constraints.include_after_last_normal = Some(rng.gen_bool(0.5));
        fname.push_str(format!("inc{}", constraints.include_after_last_normal.unwrap()).as_str());
    }

    fname
}

// wiggle for circuit fingerprinting
fn wiggle_cf_cfg(cfg: &mut Config, prob: f64, rng: &mut Xoshiro256StarStar) -> String {
    // for each change we make, we build a new filename
    let mut fname = String::new();

    let derive = cfg.derive.as_mut().unwrap();
    let machine = &mut derive.machine;
    let env = &mut derive.env;
    let constraints = &mut derive.constraints;

    // attempts
    if rng.gen_bool(prob) {
        derive.max_attempts = Some(rng.gen_range(1..1024));
        fname.push_str(format!("att{}", derive.max_attempts.unwrap()).as_str());
    }

    // states
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=5);
        let min = rng.gen_range(1..=max);
        machine.num_states = Range(min as f64, max as f64);
        fname.push_str(format!("states{}", machine.num_states).as_str());
    }

    // allow_frac_limits
    if rng.gen_bool(prob) {
        machine.allow_frac_limits = Some(rng.gen_bool(0.5));
        fname.push_str(format!("fl{}", machine.allow_frac_limits.unwrap()).as_str());
    }

    // duration ref point
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=1_000_000);
        let min = rng.gen_range(1..=max);
        machine.duration_ref_point = Some(Range(min as f64, max as f64));
        fname.push_str(format!("drp{}", machine.duration_ref_point.unwrap()).as_str());
    }
    // count ref point
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=1_000);
        let min = rng.gen_range(1..=max);
        machine.count_ref_point = Some(Range(min as f64, max as f64));
        fname.push_str(format!("crp{}", machine.count_ref_point.unwrap()).as_str());
    }
    // min action timeout
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=1_000);
        let min = rng.gen_range(1..=max);
        machine.min_action_timeout = Some(Range(min as f64, max as f64));
        fname.push_str(format!("mat{}", machine.min_action_timeout.unwrap()).as_str());
    }

    // traces
    let mut updated_traces = false;
    if rng.gen_bool(prob) {
        // clone the original traces (it's a superset)
        let mut traces = env.traces.clone();
        traces.shuffle(rng);
        let num_traces = rng.gen_range(1..=traces.len());
        traces.truncate(num_traces);
        env.traces = traces;
        fname.push_str(format!("traces{:?}", env.traces).as_str());

        updated_traces = true;
    }

    // num traces
    if updated_traces || rng.gen_bool(prob) {
        // FIXME: magic constant 14, correct for now for CF but not for other
        // types of traces
        let m = env.traces.len() * 14;
        let max = rng.gen_range(1..=m);
        let min = rng.gen_range(1..=max);
        env.num_traces = Range(min as f64, max as f64);
        fname.push_str(format!("num{}", env.num_traces).as_str());
    }

    // sim steps
    if rng.gen_bool(prob) {
        let max = rng.gen_range(1..=100_000);
        let min = rng.gen_range(1..=max);
        env.sim_steps = Range(min as f64, max as f64);
        fname.push_str(format!("steps{}", env.sim_steps).as_str());
    }

    // client load
    if rng.gen_bool(prob) {
        let max = rng.gen_range(0.0..=10.0);
        let min = rng.gen_range(0.0..=max);
        constraints.client_load = Range(round_f64(min), round_f64(max));
        fname.push_str(format!("cl{:?}", constraints.client_load).as_str());
    }
    // server load
    if rng.gen_bool(prob) {
        let max = rng.gen_range(0.0..=10.0);
        let min = rng.gen_range(0.0..=max);
        constraints.server_load = Range(round_f64(min), round_f64(max));
        fname.push_str(format!("sl{:?}", constraints.server_load).as_str());
    }
    // delay
    if constraints.delay.1 > 0.0 && rng.gen_bool(prob) {
        let max = rng.gen_range(0.0..=5.0);
        let min = rng.gen_range(0.0..=max);
        constraints.delay = Range(round_f64(min), round_f64(max));
        fname.push_str(format!("d{:?}", constraints.delay).as_str());
    }

    // client min normal packets makes no sense for CF
    // server min normal packets makes no sense for CF

    // include after last normal
    if rng.gen_bool(prob) {
        constraints.include_after_last_normal = Some(rng.gen_bool(0.5));
        fname.push_str(format!("inc{}", constraints.include_after_last_normal.unwrap()).as_str());
    }

    fname
}
