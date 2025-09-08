use std::{fs::metadata, str::FromStr};

use anyhow::{bail, Result};
use log::info;
use maybenot_gen::defense::Defense;
use maybenot_machines::{get_machine, StaticMachine};
use rand_seeder::Seeder;
use rand_xoshiro::Xoshiro256StarStar;

use crate::save_defenses;

pub fn do_fixed(
    client: Vec<String>,
    server: Vec<String>,
    output: String,
    n: Option<usize>,
    seed: Option<String>,
) -> Result<()> {
    if client.is_empty() {
        bail!("no client machines provided");
    }
    if server.is_empty() {
        bail!("no server machines provided");
    }
    if metadata(&output).is_ok() {
        bail!("output '{}' already exists", output);
    }
    let seed = seed.unwrap_or("0".to_string());
    info!("deterministic, using seed {seed}");
    let mut rng: Xoshiro256StarStar = Seeder::from(seed.clone()).make_rng();

    let n = n.unwrap_or(1);
    if n == 0 {
        bail!("n must be greater than 0");
    }

    // make a description: "fixed client [client machines], fixed server [server
    // machines], seed {}, n {}"
    let mut description = "fixed client".to_string();
    for c in &client {
        let sm = StaticMachine::from_str(c)?;
        info!("loading client machine: {sm:#?}");
        description.push_str(format!(" {c}").as_str());
    }
    description.push_str(", fixed server");
    for s in &server {
        let sm = StaticMachine::from_str(s)?;
        info!("loading server machine: {sm:#?}");
        description.push_str(format!(" {s}").as_str());
    }
    description.push_str(format!(", seed {seed}").as_str());
    description.push_str(format!(", n {n}").as_str());

    if n > 1 {
        info!("generating {n} defenses...");
    } else {
        info!("generating defense...");
    }
    let mut defenses = vec![];
    for _ in 0..n {
        let mut client_machines = vec![];
        for c in &client {
            let sm = StaticMachine::from_str(c)?;
            client_machines.extend(get_machine(&[sm], &mut rng));
        }

        let mut server_machines = vec![];
        for s in &server {
            let sm = StaticMachine::from_str(s)?;
            server_machines.extend(get_machine(&[sm], &mut rng));
        }
        let defense = Defense::new(client_machines, server_machines);
        defenses.push(defense);
    }

    save_defenses(description, &defenses, &output)
}
