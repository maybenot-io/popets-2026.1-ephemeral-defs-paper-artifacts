pub mod traces;
use anyhow::{bail, Result};
use maybenot::TriggerEvent;
use std::{fmt, time::Duration};
use traces::load_traces;

use maybenot_simulator::{
    integration::{BinDist, Integration},
    network::Network,
    queue::SimQueue,
    sim_advanced, SimulatorArgs,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::constraints::Range;

#[derive(Debug, Clone, Deserialize, PartialEq, Serialize)]
pub enum Traces {
    BigEnough,
    TorCircuit,
    DeepFingerprinting,
    GongSurakav,
    Custom { root: String, bytes: Range },
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub enum IntegrationType {
    Example,
    File { src: String },
}

/// An instance of an environment represents the concrete conditions a single
/// client-server pair is operating under. Under these conditions, we later
/// define defenses.
#[derive(Debug, Clone)]
pub struct Environment {
    pub description: String,
    pub network: Network,
    pub traces: Vec<SimQueue>,
    pub trace_durations: Vec<Vec<Duration>>,
    pub integration_type: Option<IntegrationType>,
    pub sim_args: SimulatorArgs,
}

impl Environment {
    #[allow(clippy::too_many_arguments)]
    pub fn new<R: RngCore>(
        traces_kind: &[Traces],
        num_traces: usize,
        max_sim_steps: usize,
        network: Network,
        integration: Option<IntegrationType>,
        max_padding_frac_client: f64,
        max_padding_frac_server: f64,
        max_blocking_frac_client: f64,
        max_blocking_frac_server: f64,
        rng: &mut R,
    ) -> Result<Self> {
        // get integration
        let (client_integration, server_integration) = match integration {
            Some(IntegrationType::Example) => (get_example_client(), get_example_server()),
            Some(IntegrationType::File { ref src }) => {
                let (c, s) = integration_from_file(src)?;
                (Some(c), Some(s))
            }
            None => (None, None),
        };

        // load traces
        let traces = load_traces(
            traces_kind,
            num_traces,
            network,
            &client_integration,
            &server_integration,
            rng,
        )?;

        // one instance of simulator arguments for this environment
        let mut args = SimulatorArgs::new(network, max_sim_steps, false);
        args.max_sim_iterations = max_sim_steps;
        args.client_integration = client_integration;
        args.server_integration = server_integration;
        args.max_padding_frac_client = max_padding_frac_client;
        args.max_padding_frac_server = max_padding_frac_server;
        args.max_blocking_frac_client = max_blocking_frac_client;
        args.max_blocking_frac_server = max_blocking_frac_server;

        // Compute relative durations for every sent packet for all traces,
        // once, for later use in computing constraints related to time. We get
        // the relative durations (from start) of all the packets given the
        // simulated network and integration delays. Here we use the maximum sim
        // step and no machines, so we get more or equal number of packets as
        // when machines are running with the same simulator args.
        args.insecure_rng_seed = Some(rng.next_u64());
        let trace_durations = traces
            .iter()
            .map(|trace| {
                let trace = sim_advanced(&[], &[], &mut trace.clone(), &args);
                let starting_time = trace[0].time;
                trace
                    .iter()
                    .filter(|event| matches!(event.event, TriggerEvent::TunnelSent) && event.client)
                    .map(|event| event.time - starting_time)
                    .collect()
            })
            .collect();
        // fresh seed
        args.insecure_rng_seed = Some(rng.next_u64());

        Ok(Self {
            network,
            traces,
            trace_durations,
            integration_type: integration,
            description: format!("{traces_kind:?}"),
            sim_args: args,
        })
    }
}

impl fmt::Display for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sim_args.client_integration.is_some() && self.sim_args.server_integration.is_some()
        {
            write!(
                f,
                "Environment {{{}, {}, {:?} }}",
                self.description,
                self.network,
                self.integration_type.clone().unwrap()
            )
        } else {
            write!(f, "Environment {{{}, {} }}", self.description, self.network,)
        }
    }
}

pub fn get_example_client() -> Option<Integration> {
    Some(Integration {
        action_delay: BinDist::new(GENERIC_SMALL_INTEGRATION_DELAY).unwrap(),
        reporting_delay: BinDist::new(GENERIC_SMALL_INTEGRATION_DELAY).unwrap(),
        trigger_delay: BinDist::new(GENERIC_SMALL_INTEGRATION_DELAY).unwrap(),
    })
}

pub fn get_example_server() -> Option<Integration> {
    Some(Integration {
        action_delay: BinDist::new(GENERIC_SMALL_INTEGRATION_DELAY).unwrap(),
        reporting_delay: BinDist::new(GENERIC_SMALL_INTEGRATION_DELAY).unwrap(),
        trigger_delay: BinDist::new(GENERIC_SMALL_INTEGRATION_DELAY).unwrap(),
    })
}

/// A generic small integration delay for testing purposes.
const GENERIC_SMALL_INTEGRATION_DELAY: &str = r#"
{
    "(0.0, 0.0)": 0.45,
    "(0.0, 1.0)": 0.40,
    "(1.0, 2.0)": 0.10,
    "(2.0, 3.0)": 0.05
}
"#;

pub fn integration_from_file(fname: &str) -> Result<(Integration, Integration)> {
    let contents = std::fs::read_to_string(fname)?;
    // check that it contains at least six lines
    let lines: Vec<&str> = contents.lines().collect();
    if lines.len() < 6 {
        bail!("Integration file must contain at least six lines");
    }
    let client_action_delay = match BinDist::new(lines[0]) {
        Ok(dist) => dist,
        Err(e) => bail!("Error parsing client action delay: {}", e),
    };
    let client_reporting_delay = match BinDist::new(lines[1]) {
        Ok(dist) => dist,
        Err(e) => bail!("Error parsing client reporting delay: {}", e),
    };
    let client_trigger_delay = match BinDist::new(lines[2]) {
        Ok(dist) => dist,
        Err(e) => bail!("Error parsing client trigger delay: {}", e),
    };

    let server_action_delay = match BinDist::new(lines[3]) {
        Ok(dist) => dist,
        Err(e) => bail!("Error parsing server action delay: {}", e),
    };
    let server_reporting_delay = match BinDist::new(lines[4]) {
        Ok(dist) => dist,
        Err(e) => bail!("Error parsing server reporting delay: {}", e),
    };
    let server_trigger_delay = match BinDist::new(lines[5]) {
        Ok(dist) => dist,
        Err(e) => bail!("Error parsing server trigger delay: {}", e),
    };

    Ok((
        Integration {
            action_delay: client_action_delay,
            reporting_delay: client_reporting_delay,
            trigger_delay: client_trigger_delay,
        },
        Integration {
            action_delay: server_action_delay,
            reporting_delay: server_reporting_delay,
            trigger_delay: server_trigger_delay,
        },
    ))
}
