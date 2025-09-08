pub mod combo;
pub mod config;
pub mod derive;
pub mod eval;
pub mod fixed;
pub mod print;
pub mod search;
pub mod sim;
pub mod tune_rng;

use anyhow::Result;

use clap::{Parser, Subcommand};
use combo::do_combo;
use config::read_cfg;
use derive::do_derive;
use env_logger::{Builder, Env, Target};
use eval::do_eval;
use fixed::do_fixed;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use log::{error, info};
use maybenot_gen::defense::Defense;
use print::do_print;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use search::do_search;
use serde::{Deserialize, Serialize};
use sim::do_sim;
use std::{
    collections::HashMap,
    fs::{remove_dir_all, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    process::exit,
};
use tune_rng::do_tune_rng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, arg_required_else_help = true)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// derive a defense from a seed and configuration
    Derive {
        /// seed for deterministic defense derivation
        #[arg(short, long)]
        seed: String,
        /// config path (TOML)
        #[arg(short, long)]
        config: String,
        /// verbose flag output, prints the full machine definitions
        #[arg(short, long)]
        verbose: bool,
    },
    /// search for defenses using metaheuristics
    Search {
        /// description of the search
        #[arg(short, long)]
        description: String,
        /// output path to write defenses to
        #[arg(short, long)]
        output: String,
        /// config path (TOML)
        #[arg(short, long)]
        config: String,
        /// optional number to search for, overrides config n
        #[arg(short, long)]
        n: Option<usize>,
        /// optional seed for deterministic search, overrides config seed
        #[arg(short, long)]
        seed: Option<u64>,
        /// optional max duration in seconds for the search, overrides config max_duration_sec
        #[arg(short, long)]
        max_duration_sec: Option<usize>,
    },
    /// create defenses by combining machines
    Combo {
        /// path to defenses to read machines from
        #[arg(short, long)]
        input: Vec<String>,
        /// output path to write defenses to
        #[arg(short, long)]
        output: String,
        /// config path (TOML)
        #[arg(short, long)]
        config: String,
        /// optional number of output defenses, overrides config n
        #[arg(short, long)]
        n: Option<usize>,
        /// optional max number of machines per side in defenses, overrides config height
        height: Option<usize>,
        /// optional seed, overrides config seed
        #[arg(short, long)]
        seed: Option<u64>,
    },
    /// simulate defenses on a dataset
    Sim {
        /// path to defenses
        #[arg(short, long)]
        input: Vec<String>,
        /// output path to write dataset to
        #[arg(short, long)]
        output: String,
        /// config path (TOML)
        #[arg(short, long)]
        config: String,
        /// optional number of defenses to take, overrides config take_defenses
        #[arg(short, long)]
        take: Option<usize>,
        /// optional seed, overrides config seed
        #[arg(short, long)]
        seed: Option<u64>,
        /// flag to run evaluation on the dataset after simulation
        #[arg(short, long, action)]
        eval: bool,
        /// optional flag to perform 10-fold cross-validation if eval is true
        #[arg(short, long, action)]
        fold: bool,
    },
    /// evaluate a defended dataset
    Eval {
        /// path to dataset
        #[arg(short, long)]
        input: String,
        /// config path (TOML)
        #[arg(short, long)]
        config: String,
        /// fold to evaluate, optional
        #[arg(short, long)]
        fold: Option<usize>,
    },
    /// print a defense or a machine, either from a file or from stdin
    Print {
        /// input path to defenses, optional
        #[arg(short, long)]
        input: Option<String>,
        /// optional defense index to print
        #[arg(short, long)]
        n: Option<usize>,
        /// output path to write visualization to, optional
        #[arg(short, long)]
        output: Option<String>,
    },
    /// create a defense based on static (fixed) machines
    Fixed {
        /// one or more client static machines
        #[arg(short, long)]
        client: Vec<String>,
        /// one or more server static machines
        #[arg(short, long)]
        server: Vec<String>,
        /// output path to write deck to
        #[arg(short, long)]
        output: String,
        /// number of defenses to generate (optional)
        #[arg(short, long)]
        n: Option<usize>,
        /// seed for determinism (optional)
        #[arg(short, long)]
        seed: Option<String>,
    },
    /// tune config by randomly replacing values to search for better defenses
    Tune {
        /// config path (TOML)
        #[arg(short, long)]
        config: String,
        /// probability to change each config value (set low)
        #[arg(short, long)]
        probability: f64,
        /// temporary output path to write to (ramdisk good)
        #[arg(short, long)]
        output: String,
        /// optional number of defenses to search for, overrides config n
        #[arg(short, long)]
        n: Option<usize>,
        /// optional max duration in seconds for each search, overrides config max_duration_sec
        #[arg(short, long)]
        max_duration_sec: Option<usize>,
        /// optional seed for deterministic tuning, overrides config seed
        #[arg(short, long)]
        seed: Option<u64>,
    },
}

fn main() {
    let mut builder = Builder::from_env(Env::default().default_filter_or("info"));
    builder.target(Target::Stdout);
    builder.init();

    if let Err(e) = do_main() {
        error!("error: {e}");
        exit(1);
    }
}

fn do_main() -> Result<()> {
    let args = Args::parse();
    match args.command.unwrap() {
        Commands::Derive {
            seed,
            config,
            verbose,
        } => do_derive(&read_cfg(&config)?, seed, verbose)?,
        Commands::Combo {
            input,
            output,
            config,
            n,
            height,
            seed,
        } => do_combo(input, output, &read_cfg(&config)?, n, height, seed)?,
        Commands::Search {
            description,
            output,
            config,
            n,
            seed,
            max_duration_sec,
        } => do_search(
            description,
            &read_cfg(&config)?,
            output,
            n,
            max_duration_sec,
            seed,
        )?,
        Commands::Sim {
            input,
            output,
            config,
            take,
            seed,
            eval,
            fold,
        } => {
            if eval {
                info!("running simulation and evaluation");
            }

            let cfg = read_cfg(&config)?;
            do_sim(&cfg, input, output.clone(), take, seed)?;

            if eval {
                let sim = cfg.clone().sim.unwrap();
                if sim.tunable_defense_limits.is_none() {
                    if fold {
                        for f in 0..10 {
                            do_eval(&cfg, output.clone(), Some(f))?;
                        }
                    } else {
                        do_eval(&cfg, output.clone(), None)?;
                    }
                } else {
                    let limits = sim.tunable_defense_limits.as_ref().unwrap();
                    for limit in limits.iter() {
                        let output = Path::new(&output).join(format!("limit-{limit}"));
                        if fold {
                            for f in 0..10 {
                                do_eval(&cfg, output.to_str().unwrap().to_owned(), Some(f))?;
                            }
                        } else {
                            do_eval(&cfg, output.to_str().unwrap().to_owned(), None)?;
                        }
                    }
                }
                // delete the simulated dataset folder in output
                remove_dir_all(output)?;
            }
        }
        Commands::Eval {
            input,
            config,
            fold,
        } => do_eval(&read_cfg(&config)?, input, fold)?,
        Commands::Print { input, n, output } => do_print(input, n, output)?,
        Commands::Fixed {
            client,
            server,
            output,
            n,
            seed,
        } => do_fixed(client, server, output, n, seed)?,
        Commands::Tune {
            config,
            probability,
            output,
            n,
            max_duration_sec,
            seed,
        } => do_tune_rng(
            &read_cfg(&config)?,
            probability,
            output,
            n,
            max_duration_sec,
            seed,
        )?,
    }
    Ok(())
}

fn get_progress_style() -> ProgressStyle {
    ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise:.green}] [{eta_precise:.cyan}] ({percent:.bold}%) [{bar:50.cyan/blue}] {pos}/{human_len} {msg:.magenta}",
    )
    .unwrap().progress_chars("█░")
}

// read dataset into a vector of (class, filename, path to content)
fn read_dataset(root: &Path) -> Vec<(usize, String, String)> {
    let mut dataset: Vec<(usize, String, String)> = vec![];

    if root.is_dir() {
        relative_recursive_read(root, None, &mut dataset);
    }

    dataset
}

// turn dataset into a map of (class+filename) -> path to content
fn make_dataset_map(dataset: &[(usize, String, String)]) -> HashMap<String, String> {
    let mut dataset_map = HashMap::new();
    for (relative, fname, path) in dataset.iter() {
        dataset_map.insert(format!("{relative}+{fname}"), path.to_string());
    }
    dataset_map
}

fn get_trace_content(path: &String) -> String {
    std::fs::read_to_string(path).unwrap()
}

fn relative_recursive_read(
    root: &Path,
    class: Option<usize>,
    dataset: &mut Vec<(usize, String, String)>,
) {
    for entry in std::fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            // the class is the last part of the path
            let class = class.unwrap_or_else(|| {
                path.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse::<usize>()
                    .unwrap()
            });
            relative_recursive_read(path.as_path(), Some(class), dataset);
        } else {
            // if the path ends with .log, it's a trace, read it
            if path.to_str().unwrap().ends_with(".log") {
                if class.is_none() {
                    panic!("trace file found in root directory, but no class specified");
                }
                let fname = path.file_name().unwrap().to_str().unwrap().to_string();
                dataset.push((class.unwrap(), fname, path.to_str().unwrap().to_string()));
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Defenses {
    description: String,
    defenses: Vec<Defense>,
}

fn save_defenses(description: String, defenses: &[Defense], filename: &str) -> Result<()> {
    // sort by defense.id()
    let mut defenses = defenses.to_vec();
    defenses.sort_by(|a, b| a.id().cmp(b.id()));

    // from description, strip all new lines
    let d = description.replace("\n", " ");

    let defenses = Defenses {
        description: d,
        defenses,
    };

    // iterate over defenses and serialize them
    let mut res: Vec<String> = defenses
        .defenses
        .par_iter()
        .progress_with_style(get_progress_style())
        .map(|defense| serde_json::to_string(defense).unwrap())
        .collect();

    // sort res for deterministic order
    res.sort();

    let file = File::create(filename)?;

    // as the first line, write the description
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", defenses.description)?;
    for line in res {
        writeln!(writer, "{line}")?;
    }

    info!("saved defenses to {filename}");
    Ok(())
}

fn load_defenses(filename: &str) -> Result<Defenses> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // read all lines into a vector
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

    // read the first line as the description
    let description = lines[0].clone();

    // read the rest of the lines as defenses, in parallel, dealing with errors
    let defenses: Vec<Option<Defense>> = lines[1..]
        .par_iter()
        .progress_with_style(get_progress_style())
        .map(|line| serde_json::from_str(line).ok())
        .collect();

    // if any none values are found, return an error
    if defenses.iter().any(|d| d.is_none()) {
        return Err(anyhow::anyhow!("error reading defenses"));
    }

    // unwrap the defenses
    let defenses: Vec<Defense> = defenses.into_iter().flatten().collect();

    Ok(Defenses {
        description,
        defenses,
    })
}
