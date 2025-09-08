use std::{
    io::{stdin, Read},
    str::FromStr,
};

use anyhow::{bail, Result};
use maybenot::Machine;

use crate::load_defenses;

pub fn do_print(input: Option<String>, n: Option<usize>, output: Option<String>) -> Result<()> {
    if output.is_some() {
        bail!("output file not yet supported");
    }

    // if input it Some, use that, otherwise read from stdin until EOF
    let input = match input {
        Some(input) => input,
        None => {
            let mut input = String::new();
            stdin().read_to_string(&mut input)?;
            input.trim().to_string()
        }
    };

    // attempt to parse input first as a machine, then as a defense, then bail
    // if neither
    if let Ok(m) = Machine::from_str(&input) {
        print_machine(m);
        return Ok(());
    };

    if let Ok(defenses) = load_defenses(&input) {
        if let Some(n) = n {
            if n >= defenses.defenses.len() {
                bail!("defense index out of bounds");
            }
            println!("{}", defenses.defenses[n]);
        } else {
            for d in defenses.defenses {
                println!("{d}");
            }
        }
        return Ok(());
    };

    bail!("could not parse input as a machine or defense");
}

fn print_machine(m: Machine) {
    println!("{m}");
}
