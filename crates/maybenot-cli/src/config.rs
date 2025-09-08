use anyhow::Result;
use std::fs;

use maybenot_gen::{
    constraints::Range,
    environment::{IntegrationType, Traces},
};
use serde::{Deserialize, Serialize};
use toml::from_str;

use crate::{
    combo::ComboConfig, derive::DeriveConfig, eval::EvalConfig, search::SearchConfig,
    sim::SimConfig,
};

/// parameters for the machines
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct MachineParams {
    /// the number of states for the machine
    pub num_states: Range,
    /// allow the client to cause blocking, defaults to random
    pub allow_blocking_client: Option<bool>,
    /// allow the server to cause blocking, defaults to random
    pub allow_blocking_server: Option<bool>,
    /// allow fixed padding and blocking budgets (useful for defending circuits
    /// in Tor or handshakes, less so for long-lived connections like VPNs),
    /// defaults to random
    pub allow_fixed_budget: Option<bool>,
    /// allow fractional limits for padding and blocking actions for machines,
    /// defaults to random
    pub allow_frac_limits: Option<bool>,
    /// allow expressive machines, using internal timers and counters (greatly
    /// increases search space for defenses, but machines are now technically
    /// Turing complete), defaults to random
    pub allow_expressive: Option<bool>,
    /// the point of reference for all random counts in machines (allowed
    /// padding packets and counters if applicable, and state limits), defaults
    /// to 100 if not set
    pub count_ref_point: Option<Range>,
    /// the point of reference for all random distribution parameters and
    /// maximum sampled durations in the machines, defaults to 100 ms if not set
    pub duration_ref_point: Option<Range>,
    /// the minimum action timeout on blocking and padding actions, default to 0
    /// if not set
    pub min_action_timeout: Option<Range>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct EnvParams {
    /// the traces that make up the environment
    pub traces: Vec<Traces>,
    /// the number of traces to sample
    pub num_traces: Range,
    /// sampled rtt between client and server
    pub rtt_in_ms: Range,
    /// sampled bottleneck between client and server
    pub packets_per_sec: Option<Range>,
    /// The number of simulated steps for the environment. Consists of all
    /// simulated Maybenot::TriggerEvents events at the client and server.
    pub sim_steps: Range,
    /// optional integration model of Maybenot to use for the simulation; be
    /// happy and ignore this if you don't know what it is
    pub integration: Option<IntegrationType>,
}

/// a shared configuration struct for all subcommands for easy chaining of the
/// same configuration file. Each subcommand checks for its required fields.
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Config {
    /// search for defenses that can be derived
    pub search: Option<SearchConfig>,
    /// derive a defense from a seed and configuration
    pub derive: Option<DeriveConfig>,
    /// create defenses by combining machines
    pub combo: Option<ComboConfig>,
    /// simulate defenses
    pub sim: Option<SimConfig>,
    /// evaluate defenses
    pub eval: Option<EvalConfig>,
}

pub fn read_cfg(path: &str) -> Result<Config> {
    let contents = fs::read_to_string(path)?;
    let cfg = from_str::<Config>(&contents)?;
    Ok(cfg)
}
