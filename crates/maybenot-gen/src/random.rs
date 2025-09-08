use enum_map::{enum_map, EnumMap};
use maybenot::{
    action::Action,
    constants::STATE_SIGNAL,
    counter::{Counter, Operation},
    dist::{Dist, DistType},
    event::Event,
    state::{State, Trans},
    Machine, Timer,
};
use petgraph::{
    algo::{connected_components, kosaraju_scc},
    Graph,
};
use rand::prelude::SliceRandom;
use rand::Rng;
use rand_core::RngCore;

use crate::constraints::Range;

const REF_DURATION_POINT: f64 = 100_000.0;
const REF_COUNT_POINT: u64 = 100;

/// Round parameters to 3 decimal places. Public since mutations may need to
/// round numbers and we want to ensure consistency.
pub fn round_f32(num: f32) -> f32 {
    const THREE_DECIMAL_PLACES: f32 = 1000.0;
    (num * THREE_DECIMAL_PLACES).round() / THREE_DECIMAL_PLACES
}
pub fn round_f64(num: f64) -> f64 {
    const THREE_DECIMAL_PLACES: f64 = 1000.0;
    (num * THREE_DECIMAL_PLACES).round() / THREE_DECIMAL_PLACES
}

/// Create a random machine with the given number of states, with the option to
/// allow blocking actions, expressive machines, fixed padding/blocking budgets,
/// and dead states. Expressive machines can use counter states, timer and
/// cancel actions, and all associated events. Fixed budgets are suitable for
/// Tor circuits and similar short-lived connections, not for long-lived VPN
/// connections. Dead states are states that may get stuck and never transition,
/// e.g., a state with that only transitions to another state on LimitReached
/// with a probability below 1.0.
///
/// The duration reference point is the maximum time in microseconds for any
/// duration, and influences the parameter selection for all distributions
/// sampling durations. If None, the default is 100_000 microseconds.
///
/// All machines are guaranteed to be strongly connected, i.e., all states can
/// reach each other with some probability.
#[allow(clippy::too_many_arguments)]
pub fn random_machine<R: RngCore>(
    num_states: usize,
    action_block: bool,
    expressive: bool,
    fixed_budget: bool,
    frac_limit: bool,
    allow_dead: bool,
    duration_point: Option<Range>,
    count_point: Option<Range>,
    min_action_timeout: Option<Range>,
    rng: &mut R,
) -> Machine {
    let allowed_padding_packets = if fixed_budget {
        let p = count_point.map_or(REF_COUNT_POINT, |range| range.sample_usize(rng) as u64);
        rng.gen_range(0..=p)
    } else {
        0
    };
    let allowed_blocked_microsec = if fixed_budget && action_block {
        let p = duration_point.map_or(REF_DURATION_POINT, |range| range.sample_f64(rng)) as u64;
        rng.gen_range(0..=p)
    } else {
        0
    };
    let max_padding_frac = if frac_limit {
        round_f64(rng.gen_range(0.0..=1.0))
    } else {
        0.0
    };
    let max_blocking_frac = if action_block && frac_limit {
        round_f64(rng.gen_range(0.0..=1.0))
    } else {
        0.0
    };
    loop {
        let states: Vec<State> = (0..num_states)
            .map(|_| {
                random_state(
                    num_states,
                    action_block,
                    expressive,
                    count_point.map_or(REF_COUNT_POINT, |range| range.sample_usize(rng) as u64),
                    duration_point.map_or(REF_DURATION_POINT, |range| range.sample_f64(rng)),
                    min_action_timeout,
                    rng,
                )
            })
            .collect();
        if check_machine_states(&states, allow_dead) {
            let m = Machine::new(
                allowed_padding_packets,
                max_padding_frac,
                allowed_blocked_microsec,
                max_blocking_frac,
                states,
            );
            if let Ok(m) = m {
                return m;
            }
        }
    }
}

pub fn random_state<R: RngCore>(
    num_states: usize,
    action_block: bool,
    expressive: bool,
    count_point: u64,
    duration_point: f64,
    min_action_timeout: Option<Range>,
    rng: &mut R,
) -> State {
    let action = if expressive {
        // bias towards having an action
        if rng.gen_bool(0.75) {
            Some(random_action(
                action_block,
                expressive,
                count_point,
                duration_point,
                rng,
            ))
        } else {
            None
        }
    } else {
        Some(random_action(
            action_block,
            expressive,
            count_point,
            duration_point,
            rng,
        ))
    };

    // enforce the minimum action timeout for blocking and padding actions
    if let Some(min_action_timeout) = min_action_timeout {
        match action {
            Some(Action::BlockOutgoing { mut timeout, .. })
            | Some(Action::SendPadding { mut timeout, .. }) => {
                let min = min_action_timeout.sample_f64(rng);
                if timeout.start < min {
                    timeout.start = min;
                }
            }
            _ => {}
        };
    };

    let counter = if expressive {
        match rng.gen_range(0..6) {
            // 50% chance of no counter
            0..=2 => (None, None),
            3 => (Some(random_counter(rng)), None),
            4 => (None, Some(random_counter(rng))),
            5 => (Some(random_counter(rng)), Some(random_counter(rng))),
            _ => unreachable!(),
        }
    } else {
        (None, None)
    };

    let action_has_limit = action.is_some()
        && match action.as_ref().unwrap() {
            Action::SendPadding { limit, .. }
            | Action::BlockOutgoing { limit, .. }
            | Action::UpdateTimer { limit, .. } => limit.is_some(),
            _ => false,
        };
    let transitions =
        random_transitions(num_states, action_block, expressive, action_has_limit, rng);

    let mut s = State::new(transitions);
    s.action = action;
    s.counter = counter;
    s
}

pub fn random_transitions<R: RngCore>(
    num_states: usize,
    blocking: bool,
    expressive: bool,
    has_limit: bool,
    rng: &mut R,
) -> EnumMap<Event, Vec<Trans>> {
    let mut map = enum_map! {_ => vec![]};

    for e in Event::iter() {
        // skip events that are not allowed/relevant
        if !blocking && (*e == Event::BlockingBegin || *e == Event::BlockingEnd) {
            // NOTE: this can be used to signal to this machine from another
            // machine triggering blocking, but we ignore that for now
            continue;
        }
        if !has_limit && *e == Event::LimitReached {
            continue;
        }
        if !expressive
            && (*e == Event::TimerBegin
                || *e == Event::TimerEnd
                || *e == Event::CounterZero
                || *e == Event::Signal)
        {
            continue;
        }

        // generate transitions, always considering the LimitReached event if
        // the action has a limit
        if rng.gen_bool(0.5) || has_limit && *e == Event::LimitReached {
            // number of transitions
            let n = rng.gen_range(1..=num_states);
            // pick n unique states to transition to TODO: if expressive, add
            // support for transitioning to STATE_SIGNAL
            let mut states = (0..num_states).collect::<Vec<_>>();
            states.shuffle(rng);
            states.truncate(n);

            // give each state a random probability, rounded using round(), in
            // total summing up to at most 1.0
            let mut prob: Vec<f32> = vec![0.0; n];
            loop {
                let mut sum = 0.0;
                for p in prob.iter_mut() {
                    *p = round_f32(rng.gen_range(0.1..=1.0));
                    sum += *p;
                }
                // normalize probabilities
                for p in prob.iter_mut() {
                    *p = round_f32(*p / sum);
                }
                sum = prob.iter().sum();
                if sum <= 1.0 {
                    break;
                }
            }

            // create transitions
            let mut t = vec![];
            for (s, p) in states.iter().zip(prob.iter()) {
                t.push(Trans(*s, *p));
            }

            // done, insert into map
            map[*e] = t;
        }
    }

    map
}

pub fn random_counter<R: RngCore>(rng: &mut R) -> Counter {
    let operation = match rng.gen_range(0..3) {
        0 => Operation::Increment,
        1 => Operation::Decrement,
        2 => Operation::Set,
        _ => unreachable!(),
    };

    match rng.gen_range(0..3) {
        0 => Counter {
            operation,
            dist: None,
            copy: false,
        },
        1 => Counter {
            operation,
            dist: Some(random_dist(REF_COUNT_POINT as f64, false, rng)),
            copy: false,
        },
        2 => Counter {
            operation,
            dist: None,
            copy: true,
        },
        _ => unreachable!(),
    }
}

pub fn random_action<R: RngCore>(
    blocking: bool,
    expressive: bool,
    count_point: u64,
    duration_point: f64,
    rng: &mut R,
) -> Action {
    if expressive && blocking {
        return match rng.gen_range(0..4) {
            0 => random_action_cancel(rng),
            1 => random_action_padding(count_point, duration_point, rng),
            2 => random_action_blocking(count_point, duration_point, expressive, rng),
            3 => random_action_timer(count_point, duration_point, rng),
            _ => unreachable!(),
        };
    }
    if expressive && !blocking {
        return match rng.gen_range(0..3) {
            0 => random_action_cancel(rng),
            1 => random_action_padding(count_point, duration_point, rng),
            2 => random_action_timer(count_point, duration_point, rng),
            _ => unreachable!(),
        };
    }
    if blocking {
        return match rng.gen_range(0..2) {
            0 => random_action_padding(count_point, duration_point, rng),
            1 => random_action_blocking(count_point, duration_point, expressive, rng),
            _ => unreachable!(),
        };
    }
    random_action_padding(count_point, duration_point, rng)
}

fn random_action_cancel<R: RngCore>(rng: &mut R) -> Action {
    match rng.gen_range(0..3) {
        0 => Action::Cancel {
            timer: Timer::Action,
        },
        1 => Action::Cancel {
            timer: Timer::Internal,
        },
        2 => Action::Cancel { timer: Timer::All },
        _ => unreachable!(),
    }
}

fn random_action_padding<R: RngCore>(count_point: u64, duration_point: f64, rng: &mut R) -> Action {
    Action::SendPadding {
        bypass: rng.gen_bool(0.5),
        replace: rng.gen_bool(0.5),
        timeout: random_timeout(duration_point, rng),
        limit: random_limit(count_point, rng),
    }
}

fn random_action_blocking<R: RngCore>(
    count_point: u64,
    duration_point: f64,
    expressive: bool,
    rng: &mut R,
) -> Action {
    Action::BlockOutgoing {
        bypass: rng.gen_bool(0.5),
        // replaceable blocking ignores limits, making it possible for machines
        // that repeatedly blocks to cause infinite blocking: this is too
        // powerful for random machines, so we disable it by default
        replace: match expressive {
            true => rng.gen_bool(0.5),
            false => false,
        },
        timeout: random_timeout(duration_point, rng),
        duration: random_timeout(duration_point, rng),
        limit: random_limit(count_point, rng),
    }
}

fn random_action_timer<R: RngCore>(count_point: u64, duration_point: f64, rng: &mut R) -> Action {
    Action::UpdateTimer {
        replace: rng.gen_bool(0.5),
        duration: random_timeout(duration_point, rng),
        limit: random_limit(count_point, rng),
    }
}

pub fn random_limit<R: RngCore>(count_point: u64, rng: &mut R) -> Option<Dist> {
    if rng.gen_bool(0.5) {
        Some(random_dist(count_point as f64, false, rng))
    } else {
        None
    }
}

pub fn random_timeout<R: RngCore>(duration_point: f64, rng: &mut R) -> Dist {
    random_dist(duration_point, true, rng)
}

pub fn random_dist<R: RngCore>(point: f64, is_timeout: bool, rng: &mut R) -> Dist {
    loop {
        let start = if rng.gen_bool(0.5) {
            round_f64(rng.gen_range(0.0..=point))
        } else {
            0.0
        };
        let max = if rng.gen_bool(0.5) {
            round_f64(rng.gen_range(start..=point))
        } else {
            point
        };
        let dist = Dist {
            start,
            max,
            dist: if is_timeout {
                random_timeout_dist_type(point, rng)
            } else {
                random_count_dist_type(point, rng)
            },
        };

        if dist.validate().is_ok() {
            return dist;
        }
    }
}

// create a random distribution type for counts based on the point of reference
fn random_count_dist_type<R: RngCore>(point: f64, rng: &mut R) -> DistType {
    match rng.gen_range(0..=5) {
        0 => {
            let x = round_f64(rng.gen_range(0.0..point));
            let y = round_f64(rng.gen_range(x..=point));
            DistType::Uniform { low: x, high: y }
        }
        1 => DistType::Binomial {
            trials: rng.gen_range(10..=((point as u64).max(11))),
            probability: round_f64(rng.gen_range::<f64, _>(0.0..=1.0).max(0.001)),
        },
        2 => DistType::Geometric {
            probability: round_f64(rng.gen_range::<f64, _>(0.0..=1.0).max(0.001)),
        },
        3 => DistType::Pareto {
            scale: round_f64(rng.gen_range::<f64, _>(point / 100.0..=point).max(0.001)),
            shape: round_f64(rng.gen_range(0.001..=10.0)),
        },
        4 => DistType::Poisson {
            lambda: round_f64(rng.gen_range(0.0..=point)),
        },
        5 => DistType::Weibull {
            scale: round_f64(rng.gen_range(0.0..=point)),
            shape: round_f64(rng.gen_range(0.5..5.0)),
        },
        _ => unreachable!(),
    }
}

// create a random distribution type for timeouts based on the point of reference
fn random_timeout_dist_type<R: RngCore>(point: f64, rng: &mut R) -> DistType {
    match rng.gen_range(0..=7) {
        0 => {
            let x = round_f64(rng.gen_range(0.0..point));
            let y = round_f64(rng.gen_range(x..=point));
            DistType::Uniform { low: x, high: y }
        }
        1 => DistType::Normal {
            mean: round_f64(rng.gen_range(0.0..=point)),
            stdev: round_f64(rng.gen_range(0.0..=point)),
        },
        2 => DistType::SkewNormal {
            location: round_f64(rng.gen_range(point * 0.5..=point * 1.5)),
            scale: round_f64(rng.gen_range(point / 100.0..=point / 10.0)),
            shape: round_f64(rng.gen_range(-5.0..=5.0)),
        },
        3 => DistType::LogNormal {
            mu: round_f64(rng.gen_range(0.0..=20.0)),
            sigma: round_f64(rng.gen_range(0.0..=1.0)),
        },
        4 => DistType::Pareto {
            scale: round_f64(rng.gen_range::<f64, _>(point / 100.0..=point).max(0.001)),
            shape: round_f64(rng.gen_range(0.001..=10.0)),
        },
        5 => DistType::Poisson {
            lambda: round_f64(rng.gen_range(0.0..=point)),
        },
        6 => DistType::Weibull {
            scale: round_f64(rng.gen_range(0.0..=point)),
            shape: round_f64(rng.gen_range(0.5..5.0)),
        },
        7 => DistType::Gamma {
            scale: round_f64(rng.gen_range(0.001..=point)),
            shape: round_f64(rng.gen_range(0.001..=10.0)),
        },
        _ => unreachable!(),
    }
}

/// Check if the machine states are valid, i.e., if they are strongly connected
/// (all states can reach each other) and have liveness (cannot get stuck in a
/// state without the possibility of transitioning out).
pub fn check_machine_states(states: &[State], allow_dead: bool) -> bool {
    check_strongly_connected(states) && (allow_dead || check_liveness(states))
}

// the minimum probability of a transition to be considered for the graph
const CONNECTED_MIN_EDGE_PROBABILITY: f32 = 0.05;

fn check_strongly_connected(states: &[State]) -> bool {
    let mut g = Graph::<usize, usize>::new();

    let mut nodes = vec![];
    for i in 0..states.len() {
        nodes.push(g.add_node(i));
    }
    for (si, state) in states.iter().enumerate() {
        let transitions = state.get_transitions();
        for (_, ts) in transitions {
            for t in ts {
                if t.1 >= CONNECTED_MIN_EDGE_PROBABILITY && t.0 != STATE_SIGNAL {
                    g.add_edge(nodes[si], nodes[t.0], 1);
                }
            }
        }
    }

    if connected_components(&g) != 1 {
        return false;
    }

    let s = kosaraju_scc(&g);
    if s.is_empty() {
        return false;
    }

    states.len() == s[0].len()
}

// the events that are relevant for liveness checking
const LIVENESS_EVENTS: [Event; 4] = [
    Event::NormalRecv,
    Event::NormalSent,
    Event::TunnelRecv,
    Event::TunnelSent,
];

fn check_liveness(states: &[State]) -> bool {
    // If a strongly connected graph based only on events that are guaranteed to
    // happen (NormalRecv, NormalSent, TunnelRecv, TunnelSent) regardless of
    // what machines are running, then the machine is alive.
    let mut g = Graph::<usize, usize>::new();

    let mut nodes = vec![];
    for i in 0..states.len() {
        nodes.push(g.add_node(i));
    }
    for (si, state) in states.iter().enumerate() {
        let transitions = state.get_transitions();
        for (event, ts) in transitions {
            if !LIVENESS_EVENTS.contains(&event) {
                continue;
            }
            for t in ts {
                if t.1 >= CONNECTED_MIN_EDGE_PROBABILITY {
                    g.add_edge(nodes[si], nodes[t.0], 1);
                }
            }
        }
    }

    if connected_components(&g) != 1 {
        return false;
    }

    let s = kosaraju_scc(&g);
    if s.is_empty() {
        return false;
    }

    states.len() == s[0].len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn print_results(mut samples: Vec<f64>) {
        let n = samples.len();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg = samples.iter().sum::<f64>() / n as f64;
        let median = samples[n / 2];
        let min = samples[0];
        let max = samples[n - 1];
        println!("Uniform: avg: {avg}, median: {median}, min: {min}, max: {max}",);
    }

    #[test]
    fn test_random_machine() {
        let mut rng = rand::thread_rng();
        let machine = random_machine(5, true, true, true, true, true, None, None, None, &mut rng);

        assert_eq!(machine.states.len(), 5);
    }

    #[test]
    fn test_dist_uniform() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Uniform {
                low: 0.0,
                high: 10.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_normal() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Normal {
                mean: 1000.0,
                stdev: 3.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_skew_normal() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::SkewNormal {
                location: 10_000.0,
                scale: 1000.0,
                shape: 5.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_log_normal() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::LogNormal {
                mu: 10.0,
                sigma: 1.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_binomial() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Binomial {
                trials: 1_000,
                probability: 0.5,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        // Binomial clearly only useful for counts
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_geometric() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Geometric { probability: 0.5 },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        // Geometric clearly only useful for counts
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_pareto() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Pareto {
                scale: 1_000.0,
                shape: 10.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_poisson() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Poisson {
                lambda: 1_000_000.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        // not super useful, close to uniform low === high
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_weibull() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Weibull {
                scale: 1_000_000.0,
                shape: 5.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_gamma() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Gamma {
                scale: 10_000.0,
                shape: 5.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        print_results(samples);
        //assert!(false);
    }

    #[test]
    fn test_dist_beta() {
        let dist = Dist {
            start: 0.0,
            max: 0.0,
            dist: DistType::Beta {
                alpha: 5.0,
                beta: 10.0,
            },
        };
        let mut rng = rand::thread_rng();
        let mut samples = vec![];
        for _ in 0..1000 {
            samples.push(dist.sample(&mut rng));
        }
        // beta is useless, remove
        print_results(samples);
        //assert!(false);
    }
}
