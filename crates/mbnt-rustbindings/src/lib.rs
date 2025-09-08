use numpy::PyArray1;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::collections::HashMap;

use maybenot_gen::dealer::Setup;

use rand_seeder::Seeder;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
mod dutils;
mod overhead;
pub use dutils::load_trace_to_string;
use dutils::{load_defenses, Defenses};
use itertools::Itertools;
use maybenot::{event::TriggerEvent, Machine};
use maybenot_gen::constraints::Range;
use maybenot_gen::dealer::{Dealer, DealerFixed, Limits};
use maybenot_simulator::queue::SimQueue;
use maybenot_simulator::{network::Network, parse_trace, sim_advanced, SimEvent, SimulatorArgs};
use overhead::compute_overheads;
use std::{path::Path, str::FromStr, time::Duration};

fn logs_at_client(trace: &Vec<SimEvent>) -> (Vec<u64>, Vec<i8>, Vec<bool>) {
    let starting_time = trace[0].time;

    let (times, events, paddings): (Vec<u64>, Vec<i8>, Vec<bool>) = trace
        .iter()
        .filter(|p| p.client)
        .map(|p| {
            (
                (p.time - starting_time).as_nanos() as u64,
                match p.event {
                    TriggerEvent::TunnelSent => 1,  // Upload -> 1
                    TriggerEvent::TunnelRecv => -1, // Download -> -1
                    _ => 0,
                },
                p.contains_padding,
            )
        })
        .multiunzip();

    (times, events, paddings)
}

fn netwk_monitor(trace: &Vec<SimEvent>) {
    let mut i: usize = 0;
    let n: usize = 1000;
    for p in trace.iter() {
        println!(
            "Time: {:?}, Event: {:?}, Client: {:?}, Padding: {:?}",
            p.time, p.event, p.client, p.contains_padding,
        );
        if i > n {
            break;
        }
        i += 1;
    }
}

fn cast_to_numpy_trace(
    times: Vec<u64>,
    events: Vec<i8>,
    paddings: Vec<bool>,
    py: Python,
) -> (
    Bound<PyArray1<u64>>,
    Bound<PyArray1<i8>>,
    Bound<PyArray1<bool>>,
) {
    let np_times = PyArray1::from_vec(py, times);
    let np_events = PyArray1::from_vec(py, events);
    let np_paddings = PyArray1::from_vec(py, paddings);

    (np_times, np_events, np_paddings)
}

fn convert_machines(machine_strs: Vec<String>) -> Vec<Machine> {
    machine_strs
        .iter()
        .map(|machine_str| Machine::from_str(machine_str).unwrap())
        .collect()
}

fn sim_def_on_trace_advanced(
    raw_trace: &str,
    machines_client: Vec<String>,
    machines_server: Vec<String>,
    network_delay_millis: u64,
    network_packets_per_second: usize,
    max_trace_length: usize,
    max_padding_frac_client: f64,
    max_padding_frac_server: f64,
    max_blocking_frac_client: f64,
    max_blocking_frac_server: f64,
    events_multiplier: usize,
    random_state: u64,
    debug: bool,
) -> (Vec<u64>, Vec<i8>, Vec<bool>) {
    // pps = 0 --> None, Network handles computing bottleneck from trace.
    let pps = (network_packets_per_second != 0).then_some(network_packets_per_second);

    let network = Network::new(Duration::from_millis(network_delay_millis), pps);
    let simulator_args = SimulatorArgs {
        network,
        max_trace_length,
        max_sim_iterations: max_trace_length * events_multiplier,
        continue_after_all_normal_packets_processed: true,
        only_client_events: true,
        only_network_activity: !debug,
        max_padding_frac_client,
        max_padding_frac_server,
        max_blocking_frac_client,
        max_blocking_frac_server,
        insecure_rng_seed: Some(random_state),
        client_integration: None,
        server_integration: None,
    };

    let machines_client = convert_machines(machines_client);
    let machines_server = convert_machines(machines_server);

    let mut input_trace: SimQueue = parse_trace(&raw_trace, network);
    let input_len = input_trace.len().clone();

    let trace: Vec<SimEvent> = sim_advanced(
        &machines_client,
        &machines_server,
        &mut input_trace,
        &simulator_args,
    );

    let client_logs = logs_at_client(&trace);

    if debug {
        netwk_monitor(&trace);
        println!("Input trace len: {input_len}");
        println!("Simulator args: {:?}", simulator_args);
    }

    client_logs
}

fn setup2pydict(py: Python, setup: &Setup) -> PyObject {
    let dict = PyDict::new(py);

    dict.set_item("max_padding_frac_client", setup.client.max_padding_frac)
        .unwrap();
    dict.set_item("max_padding_frac_server", setup.server.max_padding_frac)
        .unwrap();
    dict.set_item("max_blocking_frac_client", setup.client.max_blocking_frac)
        .unwrap();
    dict.set_item("max_blocking_frac_server", setup.server.max_blocking_frac)
        .unwrap();

    let client_machines: Vec<String> = setup
        .client
        .machines
        .iter()
        .map(|machine| machine.serialize())
        .collect();

    dict.set_item("client_machines", client_machines).unwrap();

    let server_machines: Vec<String> = setup
        .server
        .machines
        .iter()
        .map(|machine| machine.serialize())
        .collect();

    dict.set_item("server_machines", server_machines).unwrap();

    dict.into()
}

///
/// bindings for the maybenot simulator, and some laoding functions.
#[pymodule]
fn mbnt<'py>(m: Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "sim_trace_advanced")]
    fn sim_trace_advanced<'py>(
        py: Python<'py>,
        raw_trace: String,
        machines_client: Vec<String>,
        machines_server: Vec<String>,
        network_delay_millis: u64,
        network_packets_per_second: usize,
        max_trace_length: usize,
        max_padding_frac_client: f64,
        max_padding_frac_server: f64,
        max_blocking_frac_client: f64,
        max_blocking_frac_server: f64,
        events_multiplier: usize,
        random_state: u64,
        debug: bool,
    ) -> (
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let (times, events, paddings) = sim_def_on_trace_advanced(
            &raw_trace,
            machines_client,
            machines_server,
            network_delay_millis,
            network_packets_per_second,
            max_trace_length,
            max_padding_frac_client,
            max_padding_frac_server,
            max_blocking_frac_client,
            max_blocking_frac_server,
            events_multiplier,
            random_state,
            debug,
        );

        cast_to_numpy_trace(times, events, paddings, py)
    }

    #[pyfn(m)]
    #[pyo3(name = "sim_trace_from_file_advanced", signature = (
    path,
    machines_client,
    machines_server,
    network_delay_millis,
    network_packets_per_second,
    max_trace_length,
    max_padding_frac_client,
    max_padding_frac_server,
    max_blocking_frac_client,
    max_blocking_frac_server,
    events_multiplier,
    random_state,
    debug=false
))]
    fn sim_trace_from_file_advanced<'py>(
        py: Python<'py>,
        path: String,
        machines_client: Vec<String>,
        machines_server: Vec<String>,
        network_delay_millis: u64,
        network_packets_per_second: usize,
        max_trace_length: usize,
        max_padding_frac_client: f64,
        max_padding_frac_server: f64,
        max_blocking_frac_client: f64,
        max_blocking_frac_server: f64,
        events_multiplier: usize,
        random_state: u64,
        debug: bool,
    ) -> (
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let raw_trace = load_trace_to_string(&path).unwrap();
        let (times, events, paddings) = sim_def_on_trace_advanced(
            &raw_trace,
            machines_client,
            machines_server,
            network_delay_millis,
            network_packets_per_second,
            max_trace_length,
            max_padding_frac_client,
            max_padding_frac_server,
            max_blocking_frac_client,
            max_blocking_frac_server,
            events_multiplier,
            random_state,
            debug,
        );

        cast_to_numpy_trace(times, events, paddings, py)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_trace_to_numpy")]
    fn load_trace_to_np<'py>(
        py: Python<'py>,
        path: String,
        network_delay_millis: u64,
        network_packets_per_second: usize,
        max_trace_length: usize,
        events_multiplier: usize,
    ) -> (
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let raw_trace = load_trace_to_string(&path).unwrap();

        let (times, events, paddings) = sim_def_on_trace_advanced(
            &raw_trace,
            vec![],
            vec![],
            network_delay_millis,
            network_packets_per_second,
            max_trace_length,
            0.0,
            0.0,
            0.0,
            0.0,
            events_multiplier,
            0,
            false,
        );

        cast_to_numpy_trace(times, events, paddings, py)
    }

    #[pyfn(m)]
    #[pyo3(name = "load_trace_to_str")]
    fn load_trace_to_str(path: String) -> PyResult<String> {
        match load_trace_to_string(&path) {
            Ok(content) => Ok(content), // Return the file content
            Err(e) => Err(PyIOError::new_err(format!("Failed to read file: {}", e))),
        }
    }

    #[pyfn(m)]
    #[pyo3(name = "compute_overheads")]
    fn compute_overheads_(
        path_base: String,
        path_defended: String,
        max: usize,
        real_world: bool,
    ) -> PyResult<HashMap<String, f64>> {
        let path_base = Path::new(&path_base);
        let path_defended = Path::new(&path_defended);
        let dict = compute_overheads(path_base, path_defended, max, real_world);

        Ok(dict)
    }

    #[pyfn(m)]
    #[pyo3(name = "deal_machines", signature = (path, limits, n_machines, scale, seed=None))]
    fn deal_machines(
        py: Python,
        path: String,
        limits: HashMap<String, HashMap<String, (f64, f64)>>,
        n_machines: usize,
        scale: f64,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        let defenses: Defenses = load_defenses(&path).unwrap();

        let get_range = |client_server_key: &str, limit_key: &str| {
            limits
                .get(client_server_key)
                .and_then(|inner_map| inner_map.get(limit_key))
                .copied()
                .map(|(a, b)| Range(a, b))
        };

        let padding_budget = get_range("client", "padding_budget").unwrap();
        let blocking_budget = get_range("client", "blocking_budget").unwrap();
        let padding_frac = get_range("client", "padding_frac").unwrap();
        let blocking_frac = get_range("client", "blocking_frac").unwrap();
        let limits_client = Limits {
            padding_budget,
            blocking_budget,
            padding_frac,
            blocking_frac,
        };

        let padding_budget = get_range("server", "padding_budget").unwrap();
        let blocking_budget = get_range("server", "blocking_budget").unwrap();
        let padding_frac = get_range("server", "padding_frac").unwrap();
        let blocking_frac = get_range("server", "blocking_frac").unwrap();
        let limits_server = Limits {
            padding_budget,
            blocking_budget,
            padding_frac,
            blocking_frac,
        };

        let mut rng = match &seed {
            Some(seed) => Seeder::from(seed).make_rng(),
            None => Xoshiro256StarStar::from_entropy(),
        };

        let mut dealer = DealerFixed::new(
            defenses.defenses.to_vec(),
            Some(limits_client),
            Some(limits_server),
            false,
            &mut rng,
        )
        .unwrap();

        let setups: Vec<Setup> = dealer.draw_n(n_machines, scale, &mut rng).unwrap();

        let pylist: Vec<PyObject> = setups.iter().map(|setup| setup2pydict(py, setup)).collect();

        Ok(pylist.into_py(py))
    }

    Ok(())
}
