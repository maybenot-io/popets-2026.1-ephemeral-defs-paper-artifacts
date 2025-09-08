use anyhow::Result;
use rand::{Rng, RngCore};
use std::{fmt, time::Duration};

use anyhow::bail;
use maybenot::{Machine, TriggerEvent};
use maybenot_simulator::{sim_advanced, SimEvent};
use serde::{Deserialize, Serialize};

use crate::environment::Environment;

/// A range is a pair of values, a minimum and a maximum, checked inclusively.
/// Note that we need this type because toml and std::ops::Range are not
/// compatible.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Range(pub f64, pub f64);

impl Range {
    pub fn sample_f64<R: RngCore>(&self, rng: &mut R) -> f64 {
        assert!(self.0 <= self.1);
        rng.gen_range(self.0..=self.1)
    }

    pub fn sample_usize<R: RngCore>(&self, rng: &mut R) -> usize {
        assert!(self.0 <= self.1);
        rng.gen_range(self.0 as usize..=self.1 as usize)
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.0, self.1)
    }
}

/// Constraints on defenses in a given search space.
///
/// The constraints are expressed as overheads, i.e., load and delay, and
/// minimal observed normal packets.
///
/// The load is the percentage of additional (padding) packets, and the delay is
/// the fraction of duration delayed, both compared to the base case without a
/// defense. Additionally, the load is defined per side (client and server),
/// since padding can be asymmetric (just like common traffic to defend, e.g.,
/// web traffic). Delay is a single value, since padding and blocking on both
/// sides cause aggregate delays from propagated delays (in a realistic
/// simulator).
///
/// The minimal number of normal packets is a sanity check to ensure that the
/// defense does not complete block traffic or overwhelm the simulator with
/// padding packets or TriggerEvents (e.g., infinite BlockingBegin ->
/// BlockingBegin or BlockingBegin -> BlockingEnd loops). Random defenses,
/// especially with learning, get creative fast.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Constraints {
    /// Load is (#defended packets / #undefended packets) - 1. The range is
    /// checked inclusively (i.e., min <= x <= max). Disabled if both min and
    /// max are 0. Packets sent by the client.
    pub client_load: Range,
    /// Load is (#defended packets / #undefended packets) - 1. The range is
    /// checked inclusively (i.e., min <= x <= max). Disabled if both min and
    /// max are 0. Packets sent by the server.
    pub server_load: Range,
    /// Delay is (time with defense / time without defense) - 1.0, checked
    /// inclusively (i.e., min <= x <= max). Disabled if both min and max are 0.
    /// For computing the delay we find the location of the last *normal* packet
    /// in the simulated trace and compare the time it was sent with the base
    /// time for the same packet. This is needed since we simulate a subset of
    /// the trace and only care about the delay of the normal packets.
    pub delay: Range,
    /// Minimum number of normal packets sent by the client.
    pub client_min_normal_packets: Option<usize>,
    /// Minimum number of normal  packets sent by the server.
    pub server_min_normal_packets: Option<usize>,
    /// Include all events after the last normal packet in the defended trace.
    /// If false (default), all events after the last normal packet are stripped
    /// from the trace. This mimics how overheads are typically computed.
    pub include_after_last_normal: Option<bool>,
}

impl Constraints {
    pub fn new(
        client_load: Range,
        server_load: Range,
        delay: Range,
        client_min_normal_packets: Option<usize>,
        server_min_normal_packets: Option<usize>,
        strip_after_last_normal: Option<bool>,
    ) -> Self {
        Self {
            client_load,
            server_load,
            delay,
            client_min_normal_packets,
            server_min_normal_packets,
            include_after_last_normal: strip_after_last_normal,
        }
    }

    pub fn check(
        &self,
        client: &[Machine],
        server: &[Machine],
        env: &Environment,
        seed: u64,
    ) -> Result<()> {
        // flags to ignore some constraints
        let ignore_client_load = self.client_load == Range(0.0, 0.0);
        let ignore_server_load = self.server_load == Range(0.0, 0.0);
        let ignore_delay = self.delay == Range(0.0, 0.0);

        // vector of stats for each trace
        let mut client_stats = vec![Stats::new(); env.traces.len()];
        let mut server_stats = vec![Stats::new(); env.traces.len()];
        let mut undefended_duration = Duration::from_secs(0);
        let mut defended_duration = Duration::from_secs(0);

        // setup the simulator to give us all events, but with limits on
        // iterations and trace length
        let mut args = env.sim_args.clone();
        args.only_client_events = false;
        args.insecure_rng_seed = Some(seed);

        // TODO: just killed max_trace_length ... makes no sense with all
        // events, experiment with manually bolting back on ... need MIN packets?

        // compute all stats
        for (i, pq) in env.traces.iter().enumerate() {
            // sim and increment seed for each trace
            let mut trace = sim_advanced(client, server, &mut pq.clone(), &args);
            args.insecure_rng_seed = args.insecure_rng_seed.map(|s| s + 1);

            // strip all events after the last normal packet in the defended
            // trace, if requested
            if !self.include_after_last_normal.unwrap_or(false) {
                let last_normal = trace.iter().rev().position(|event| {
                    (matches!(event.event, TriggerEvent::TunnelSent)
                        || matches!(event.event, TriggerEvent::TunnelRecv))
                        && !event.contains_padding
                });
                if let Some(last_normal) = last_normal {
                    trace.truncate(trace.len() - last_normal);
                }
            }

            // compute stats for the trace
            count_events(&mut client_stats[i], &mut server_stats[i], &trace);

            // early, obvious checks (that may filter out a significant number
            // of machines, so we do them ASAP)
            if !ignore_client_load
                && client_stats[i].normal < self.client_min_normal_packets.unwrap_or(1)
            {
                bail!("too few normal client packets");
            }
            if !ignore_server_load
                && server_stats[i].normal < self.server_min_normal_packets.unwrap_or(1)
            {
                bail!("too few normal server packets");
            }

            // If a machine should produce some padding, check for at least 5
            // padding packets (arbitrary number) in the trace. This is to
            // ensure that the machine actually does something.
            if !ignore_client_load && self.client_load.0 > 0.001 && client_stats[i].padding < 5 {
                bail!("too few padding packets from client");
            }
            if !ignore_server_load && self.server_load.0 > 0.001 && server_stats[i].padding < 5 {
                bail!("too few padding packets from server");
            }

            // event sanity check: if we spend less than 20% of counted
            // TriggerEvents (note: no recv events) sending packets, something
            // is wrong
            if !ignore_client_load
                && (client_stats[i].sum_packets() as f64 / client_stats[i].len() as f64) < 0.20
            {
                bail!("too many non-packet events from client");
            }
            if !ignore_server_load
                && (server_stats[i].sum_packets() as f64 / server_stats[i].len() as f64) < 0.20
            {
                bail!("too many non-packet events from server");
            }

            // if we care about delay, compute the durations
            if !ignore_delay {
                match get_durations(&trace, &env.trace_durations[i]) {
                    (Some(defended), Some(undefended)) => {
                        if defended < undefended {
                            bail!("defended trace duration shorter than base trace, simulation error?");
                        }
                        defended_duration += defended;
                        undefended_duration += undefended;
                    }
                    _ => bail!("no normal network activity"),
                }
            }
        }

        if !ignore_client_load {
            check_load(&client_stats, self.client_load, "client")?;
        }
        if !ignore_server_load {
            check_load(&server_stats, self.server_load, "server")?;
        }
        if !ignore_delay {
            check_delay(undefended_duration, defended_duration, self.delay)?;
        }

        Ok(())
    }
}

fn check_delay(time_base: Duration, time_defended: Duration, oh: Range) -> Result<()> {
    // delay is (defended / undefended) - 1.0
    let avg_delay = (time_defended.as_secs_f64() / time_base.as_secs_f64()) - 1.0;
    if avg_delay < oh.0 {
        bail!(
            "average delay too low: got {}, expected [{}, {}]",
            avg_delay,
            oh.0,
            oh.1
        );
    }
    if avg_delay > oh.1 {
        bail!(
            "average delay too high: got {}, expected [{}, {}]",
            avg_delay,
            oh.0,
            oh.1
        );
    }
    Ok(())
}

fn check_load(stats: &[Stats], oh: Range, party: &str) -> Result<()> {
    // load is (defended / undefended) - 1.0
    let normal = stats.iter().map(|s| s.normal).sum::<usize>() as f64;
    let padding = stats.iter().map(|s| s.padding).sum::<usize>() as f64;
    let avg_load = (padding + normal) / normal - 1.0;

    if avg_load < oh.0 {
        bail!(
            "average load too low from {}: got {}, expected [{}, {}]",
            party,
            avg_load,
            oh.0,
            oh.1
        );
    }
    if avg_load > oh.1 {
        bail!(
            "average load too high from {}: got {}, expected [{}, {}]",
            party,
            avg_load,
            oh.0,
            oh.1
        );
    }

    Ok(())
}

#[derive(Clone)]
struct Stats {
    /// sum of padding packets sent (from TriggerEvent:TunnelSent with the
    /// padding flag)
    padding: usize,
    /// sum of normal packets sent (from TriggerEvent:TunnelSent without the
    /// padding flag)
    normal: usize,
    /// sum of blocking begin events
    blocking_begin: usize,
    /// sum of blocking end events
    blocking_end: usize,
    /// sum of timer begin events
    timer_begin: usize,
    /// sum of timer end events
    timer_end: usize,
}

impl Stats {
    pub fn new() -> Self {
        Self {
            padding: 0,
            normal: 0,
            blocking_begin: 0,
            blocking_end: 0,
            timer_begin: 0,
            timer_end: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.padding
            + self.normal
            + self.blocking_begin
            + self.blocking_end
            + self.timer_begin
            + self.timer_end
    }

    pub fn sum_packets(&self) -> usize {
        self.padding + self.normal
    }
}

fn count_events(client: &mut Stats, server: &mut Stats, trace: &[SimEvent]) {
    // iterate over the trace and count the events for client and server
    for event in trace {
        match event.event {
            TriggerEvent::TunnelSent => {
                if event.contains_padding {
                    if event.client {
                        client.padding += 1;
                    } else {
                        server.padding += 1;
                    }
                } else if event.client {
                    client.normal += 1;
                } else {
                    server.normal += 1;
                }
            }
            TriggerEvent::BlockingBegin { .. } => {
                if event.client {
                    client.blocking_begin += 1;
                } else {
                    server.blocking_begin += 1;
                }
            }
            TriggerEvent::BlockingEnd => {
                if event.client {
                    client.blocking_end += 1;
                } else {
                    server.blocking_end += 1;
                }
            }
            TriggerEvent::TimerBegin { .. } => {
                if event.client {
                    client.timer_begin += 1;
                } else {
                    server.timer_begin += 1;
                }
            }
            TriggerEvent::TimerEnd { .. } => {
                if event.client {
                    client.timer_end += 1;
                } else {
                    server.timer_end += 1;
                }
            }
            _ => {}
        }
    }
}

// get the base and defended durations for the trace, based on the last normal
// packet in the defended trace
fn get_durations(defended: &[SimEvent], base: &[Duration]) -> (Option<Duration>, Option<Duration>) {
    let starting_time = defended[0].time;
    // the duration of the last normal sent packet in the defended trace for the client
    let defended_duration = defended.iter().rev().find_map(|event| {
        if let TriggerEvent::TunnelSent = event.event {
            if !event.contains_padding && event.client {
                return Some(event.time - starting_time);
            }
        }
        None
    });

    // the number of normal packets in the defended trace
    let defended_normal = defended
        .iter()
        .filter(|event| {
            matches!(event.event, TriggerEvent::TunnelSent)
                && !event.contains_padding
                && event.client
        })
        .count();

    // get the base duration for the same number of normal packets
    let base_duration = base.get(defended_normal - 1).cloned();

    (defended_duration, base_duration)
}
