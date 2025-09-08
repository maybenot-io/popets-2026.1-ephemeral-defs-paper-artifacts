use anyhow::{bail, Result};
use maybenot::Machine;
use rand::{seq::SliceRandom, RngCore};
use serde::{Deserialize, Serialize};

use crate::{constraints::Range, defense::Defense};

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Setup {
    pub client: Params,
    pub server: Params,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Params {
    pub machines: Vec<Machine>,
    pub max_padding_frac: f64,
    pub max_blocking_frac: f64,
}

/// A "dealer" "draws" a "setup" for the client and the server. Each party gets
/// a list of machines, and maximum fractions of padding and blocking for their
/// instance of Maybenot. The scale parameter is used to scale the padding and
/// blocking budgets for the client and server, between (0.0, 1.0].
pub trait Dealer {
    /// draw a setup for the client and server
    fn draw<R: RngCore>(&mut self, scale: f64, rng: &mut R) -> Result<Setup>;
    /// draw n setups for the client and server
    fn draw_n<R: RngCore>(&mut self, n: usize, scale: f64, rng: &mut R) -> Result<Vec<Setup>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Limits {
    pub padding_budget: Range,
    pub blocking_budget: Range,
    pub padding_frac: Range,
    pub blocking_frac: Range,
}

/// A dealer based on a fixed list of defenses. The defenses are either drawn
/// with replacement (reused) or not. Setup fractions are sampled from the
/// optionally provided limits.
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct DealerFixed {
    defenses: Vec<Defense>,
    client_limits: Option<Limits>,
    server_limits: Option<Limits>,
    reuse: bool,
}

impl DealerFixed {
    pub fn new<R: RngCore>(
        mut defenses: Vec<Defense>,
        client_limits: Option<Limits>,
        server_limits: Option<Limits>,
        reuse: bool,
        rng: &mut R,
    ) -> Result<Self> {
        if defenses.is_empty() {
            bail!("no defenses provided");
        }
        if !reuse {
            defenses.shuffle(rng);
        }
        Ok(Self {
            defenses,
            client_limits,
            server_limits,
            reuse,
        })
    }
}

impl Dealer for DealerFixed {
    fn draw<R: RngCore>(&mut self, scale: f64, rng: &mut R) -> Result<Setup> {
        if !(0.0..=1.0).contains(&scale) {
            bail!("invalid scale");
        }
        if self.defenses.is_empty() {
            bail!("no defenses left");
        }

        let mut def = if self.reuse {
            self.defenses.choose_mut(rng).unwrap().clone()
        } else {
            self.defenses.pop().unwrap()
        };

        // sample and scale client and server machine limits
        if let Some(limits) = &self.client_limits {
            set_machine_limits(&mut def.client, limits, scale, rng);
        }
        if let Some(limits) = &self.server_limits {
            set_machine_limits(&mut def.server, limits, scale, rng);
        }

        // return the setup, sampling padding and blocking fractions for the
        // client and server framework instances if limits are set (0.0 means no
        // limit in the framework)
        Ok(Setup {
            client: Params {
                machines: def.client,
                max_padding_frac: self
                    .client_limits
                    .as_ref()
                    .map_or(0.0, |limits| limits.padding_frac.sample_f64(rng) * scale),
                max_blocking_frac: self
                    .client_limits
                    .as_ref()
                    .map_or(0.0, |limits| limits.blocking_frac.sample_f64(rng) * scale),
            },
            server: Params {
                machines: def.server,
                max_padding_frac: self
                    .server_limits
                    .as_ref()
                    .map_or(0.0, |limits| limits.padding_frac.sample_f64(rng) * scale),
                max_blocking_frac: self
                    .server_limits
                    .as_ref()
                    .map_or(0.0, |limits| limits.blocking_frac.sample_f64(rng) * scale),
            },
        })
    }

    fn draw_n<R: RngCore>(&mut self, n: usize, scale: f64, rng: &mut R) -> Result<Vec<Setup>> {
        if !(0.0..=1.0).contains(&scale) {
            bail!("invalid scale");
        }
        if self.defenses.is_empty() {
            bail!("no defenses left");
        }
        if !self.reuse && n > self.defenses.len() {
            bail!("not enough defenses left");
        }

        let mut setups = Vec::with_capacity(n);
        for _ in 0..n {
            setups.push(self.draw(scale, rng).unwrap());
        }
        Ok(setups)
    }

    fn len(&self) -> usize {
        self.defenses.len()
    }

    fn is_empty(&self) -> bool {
        self.defenses.is_empty()
    }
}

fn set_machine_limits<R: RngCore>(
    machines: &mut [Machine],
    limits: &Limits,
    scale: f64,
    rng: &mut R,
) {
    let padding_budget = limits.padding_budget.sample_f64(rng) * scale;
    let blocking_budget = limits.blocking_budget.sample_f64(rng) * scale;

    let n_machines_with_padding_budget = machines
        .iter()
        .filter(|m| m.allowed_padding_packets > 0)
        .count() as f64;
    let n_machines_with_blocking_budget = machines
        .iter()
        .filter(|m| m.allowed_blocked_microsec > 0)
        .count() as f64;

    for m in machines.iter_mut() {
        // we use framework fraction limits and disable machine-specific ones:
        // this is more chaotic with several machines, because interactions are
        // harder to predict
        m.max_padding_frac = 0.0;
        m.max_blocking_frac = 0.0;

        // don't scale if the machine doesn't have a budget: if there's a
        // budget, split it evenly among the machines
        if m.allowed_padding_packets > 0 {
            m.allowed_padding_packets = (padding_budget / n_machines_with_padding_budget) as u64;
        }
        if m.allowed_blocked_microsec > 0 {
            m.allowed_blocked_microsec = (blocking_budget / n_machines_with_blocking_budget) as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::random_machine;

    #[test]
    fn test_static_dealer() {
        let mut rng = rand::thread_rng();

        #[derive(Debug, Deserialize)]
        struct LimitsWrapper {
            client: Option<Limits>,
            server: Option<Limits>,
        }

        let l: LimitsWrapper = serde_json::from_str(
            r#"{
            "client": {
                "padding_budget": [0.0, 100.0],
                "blocking_budget": [0.0, 100.0],
                "padding_frac": [0.0, 1.0],
                "blocking_frac": [0.0, 1.0]
            },
            "server": {
                "padding_budget": [0.0, 100.0],
                "blocking_budget": [0.0, 100.0],
                "padding_frac": [0.0, 1.0],
                "blocking_frac": [0.0, 1.0]
            }
        }"#,
        )
        .unwrap();

        let defenses = [Defense::new(
            [random_machine(
                3, false, false, false, false, false, None, None, None, &mut rng,
            )]
            .to_vec(),
            [random_machine(
                3, false, false, false, false, false, None, None, None, &mut rng,
            )]
            .to_vec(),
        )]
        .to_vec();
        let mut dealer = DealerFixed::new(defenses, l.client, l.server, false, &mut rng).unwrap();

        let setup = dealer.draw(0.5, &mut rng).unwrap();
        assert_eq!(setup.client.machines.len(), 1);
        assert_eq!(setup.server.machines.len(), 1);
        assert!(setup.client.max_padding_frac <= 0.5);
        assert!(setup.client.max_blocking_frac <= 0.5);
        assert!(setup.server.max_padding_frac <= 0.5);
        assert!(setup.server.max_blocking_frac <= 0.5);
    }
}
