use std::str::FromStr;

use anyhow::{bail, Result};
use break_pad::{break_pad_client, break_pad_server};
use enum_map::enum_map;

use front::front;
use interspace::{interspace_client, interspace_server};
use maybenot::{state::State, Machine};
use rand::prelude::SliceRandom;
use rand_core::RngCore;
use regulator::{regulator_client, regulator_server};
use scrambler::{scrambler_client, scrambler_server};
use serde::{Deserialize, Serialize};
use tamaraw::{tamaraw, tamaraw_client, tamaraw_server};

pub mod break_pad;
pub mod front;
pub mod interspace;
pub mod regulator;
pub mod scrambler;
pub mod tamaraw;

/// Static machines are hardcoded machines. They are used for testing, as a
/// starting point for evolving machines, and as part of defenses.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum StaticMachine {
    /// A minimal machine that does nothing but is still a valid machine.
    NoOp,
    /// Randomly select one of the above machines, excluding "NoOp".
    Random,
    RegulatorClient {
        u: f64,
        c: f64,
    },
    RegulatorServer {
        r: f64,
        d: f64,
        t: f64,
        n: f64,
        b: usize,
    },
    TamarawClient,
    TamarawServer,
    Tamaraw {
        /// padding rate in s/packet
        p: f64,
        /// the duration (in microseconds) that stops Tamaraw if no normal
        /// packet has been sent within
        stop_window: f64,
    },
    InterspaceClient,
    InterspaceServer,
    Front {
        padding_budget_max: u32,
        window_min: f64,
        window_max: f64,
        num_states: usize,
    },
    BreakPadClient,
    BreakPadServer,
    ScramblerClient,
    ScramblerServer {
        interval: f64,
        min_count: f64,
        min_trail: f64,
        max_trail: f64,
    },
}

impl FromStr for StaticMachine {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        if s.contains("tamaraw ") {
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() != 3 {
                bail!("invalid tamaraw defense: {}", s);
            }
            let p = parts[1].parse::<f64>().map_err(|_| {
                anyhow::anyhow!("invalid tamaraw defense parameter 'p': {}", parts[1])
            })?;
            let stop_window = parts[2].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid tamaraw defense parameter 'stop_window': {}",
                    parts[2]
                )
            })?;
            return Ok(StaticMachine::Tamaraw { p, stop_window });
        }

        if s.contains("regulator_client ") {
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() != 3 {
                bail!("invalid regulator client defense: {}", s);
            }
            let u = parts[1].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid regulator client defense parameter 'u': {}",
                    parts[1]
                )
            })?;
            let c = parts[2].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid regulator client defense parameter 'c': {}",
                    parts[2]
                )
            })?;
            return Ok(StaticMachine::RegulatorClient { u, c });
        }
        if s.contains("regulator_server ") {
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() != 6 {
                bail!("invalid regulator server defense: {}", s);
            }
            let r = parts[1].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid regulator server defense parameter 'r': {}",
                    parts[1]
                )
            })?;
            let d = parts[2].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid regulator server defense parameter 'd': {}",
                    parts[2]
                )
            })?;
            let t = parts[3].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid regulator server defense parameter 't': {}",
                    parts[3]
                )
            })?;
            let n = parts[4].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid regulator server defense parameter 'n': {}",
                    parts[4]
                )
            })?;
            let b = parts[5].parse::<usize>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid regulator server defense parameter 'b': {}",
                    parts[5]
                )
            })?;
            return Ok(StaticMachine::RegulatorServer { r, d, t, n, b });
        }

        if s.contains("front ") {
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() != 5 {
                bail!("invalid front defense: {}", s);
            }
            let padding_budget_max = parts[1].parse::<u32>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid front defense parameter 'padding_budget_max': {}",
                    parts[1]
                )
            })?;
            let window_min = parts[2].parse::<f64>().map_err(|_| {
                anyhow::anyhow!("invalid front defense parameter 'window_min': {}", parts[2])
            })?;
            let window_max = parts[3].parse::<f64>().map_err(|_| {
                anyhow::anyhow!("invalid front defense parameter 'window_max': {}", parts[3])
            })?;
            let num_states = parts[4].parse::<usize>().map_err(|_| {
                anyhow::anyhow!("invalid front defense parameter 'num_states': {}", parts[4])
            })?;
            return Ok(StaticMachine::Front {
                padding_budget_max,
                window_min,
                window_max,
                num_states,
            });
        }

        if s.contains("scrambler_server") {
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() != 5 {
                bail!("invalid scrambler server defense: {}", s);
            }
            let interval = parts[1].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid scrambler server defense parameter 'interval': {}",
                    parts[1]
                )
            })?;
            let min_count = parts[2].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid scrambler server defense parameter 'min_count': {}",
                    parts[2]
                )
            })?;
            let min_trail = parts[3].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid scrambler server defense parameter 'min_trail': {}",
                    parts[3]
                )
            })?;
            let max_trail = parts[4].parse::<f64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid scrambler server defense parameter 'max_trail': {}",
                    parts[4]
                )
            })?;
            return Ok(StaticMachine::ScramblerServer {
                interval,
                min_count,
                min_trail,
                max_trail,
            });
        }

        match s {
            "noop" => Ok(StaticMachine::NoOp),
            "random" => Ok(StaticMachine::Random),
            "tamaraw_client" => Ok(StaticMachine::TamarawClient),
            "tamaraw_server" => Ok(StaticMachine::TamarawServer),
            "interspace_client" => Ok(StaticMachine::InterspaceClient),
            "interspace_server" => Ok(StaticMachine::InterspaceServer),
            "break_pad_client" => Ok(StaticMachine::BreakPadClient),
            "break_pad_server" => Ok(StaticMachine::BreakPadServer),
            "scrambler_client" => Ok(StaticMachine::ScramblerClient),
            _ => bail!("invalid static machine: {}", s),
        }
    }
}

/// Get one or more machines from a list of static machines.
pub fn get_machine<R: RngCore>(s: &[StaticMachine], rng: &mut R) -> Vec<Machine> {
    let mut machines = vec![];

    for m in s {
        match m {
            StaticMachine::NoOp => machines.push(no_op_machine()),
            StaticMachine::Random => {
                // Since we removed the machines, Random now selects from remaining machines
                let selected = [StaticMachine::NoOp].choose(rng).unwrap();
                machines.extend(get_machine(&[*selected], rng))
            }
            StaticMachine::RegulatorClient { u, c } => machines.extend(regulator_client(*u, *c)),
            StaticMachine::RegulatorServer { r, d, t, n, b } => {
                machines.extend(regulator_server(*r, *d, *t, *n, *b))
            }
            StaticMachine::TamarawClient => machines.extend(tamaraw_client()),
            StaticMachine::TamarawServer => machines.extend(tamaraw_server()),
            StaticMachine::Tamaraw { p, stop_window } => machines.extend(tamaraw(*p, *stop_window)),
            StaticMachine::InterspaceClient => machines.extend(interspace_client(rng)),
            StaticMachine::InterspaceServer => machines.extend(interspace_server(rng)),
            StaticMachine::Front {
                padding_budget_max,
                window_min,
                window_max,
                num_states,
            } => machines.extend(front(
                *padding_budget_max,
                *window_min,
                *window_max,
                *num_states,
                rng,
            )),
            StaticMachine::BreakPadClient => machines.extend(break_pad_client()),
            StaticMachine::BreakPadServer => machines.extend(break_pad_server()),
            StaticMachine::ScramblerClient => machines.extend(scrambler_client()),
            StaticMachine::ScramblerServer {
                interval,
                min_count,
                min_trail,
                max_trail,
            } => machines.extend(scrambler_server(
                *interval, *min_count, *min_trail, *max_trail,
            )),
        };
    }

    machines
}

fn no_op_machine() -> Machine {
    let s0 = State::new(enum_map! {
        _ => vec![],
    });
    Machine::new(0, 0.0, 0, 0.0, vec![s0]).unwrap()
}

// serialization tests for hardcoded machines handed out at different times
#[cfg(test)]
mod tests {
    use maybenot::{
        action::Action,
        dist::{Dist, DistType},
        event::Event,
        state::Trans,
    };

    use crate::*;

    #[test]
    fn test_no_op() {
        assert!(no_op_machine().validate().is_ok());
        assert_eq!(no_op_machine().serialize(), "02eNpjYEAHjOgCAAA0AAI=")
    }

    #[test]
    fn test_example_machine() {
        let mut states = vec![];

        let start_state = State::new(enum_map! {
            Event::TunnelRecv => vec![Trans(1, 1.0)],
            _ => vec![],
        });
        states.push(start_state);

        let mut padding_state = State::new(enum_map! {
            Event::PaddingSent => vec![Trans(0, 1.0)],
            _ => vec![],
        });
        padding_state.action = Some(Action::SendPadding {
            bypass: false,
            replace: false,
            timeout: Dist {
                dist: DistType::Uniform {
                    low: 0.0,
                    high: 0.0,
                },
                start: 5_000_000.0,
                max: 0.0,
            },
            limit: None,
        });
        states.push(padding_state);

        let m = Machine::new(u64::MAX, 1.0, 0, 0.0, states).unwrap();
        assert_eq!(
            m.serialize(),
            "02eNpti1EJACAQQzcjWEjMYCEjGsUCojiUg+Pex2CPbe0HxCz4JCVJoJvF7SEjt+qUtnY+gDUNIg=="
        );
    }

    #[test]
    fn test_ping_1s_loop_machine() {
        let mut states = vec![];

        let start_state = State::new(enum_map! {
            Event::NormalSent => vec![Trans(1, 1.0)],
            Event::NormalRecv => vec![Trans(1, 1.0)],
            _ => vec![],
        });
        states.push(start_state);

        let mut padding_state = State::new(enum_map! {
            Event::PaddingSent => vec![Trans(1, 1.0)],
            _ => vec![],
        });
        padding_state.action = Some(Action::SendPadding {
            bypass: false,
            replace: false,
            timeout: Dist {
                dist: DistType::Uniform {
                    low: 0.0,
                    high: 0.0,
                },
                start: 1_000_000.0,
                max: 0.0,
            },
            limit: None,
        });
        states.push(padding_state);

        let m = Machine::new(u64::MAX, 1.0, 0, 0.0, states).unwrap();
        assert_eq!(
            m.serialize(),
            "02eNpty8sJACAMA9DEgVyhuFkPDuwC4o8KpfRBoQlkLoNnCL5yjiSg4h5zY0p7baEK2w33hw3i"
        );
    }
}
