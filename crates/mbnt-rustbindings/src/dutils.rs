use anyhow::Result;
use maybenot_gen::defense::Defense;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

// read dataset into a vector of (class, filename, path to content)
pub fn read_dataset(root: &Path) -> Vec<(usize, String, String)> {
    let mut dataset: Vec<(usize, String, String)> = vec![];

    if root.is_dir() {
        relative_recursive_read(root, None, &mut dataset);
    }

    dataset
}

pub fn load_trace_to_string(path: &str) -> Result<String, io::Error> {
    std::fs::read_to_string(path)
}

// turn dataset into a map of (class+filename) -> path to content
pub fn make_dataset_map(dataset: &[(usize, String, String)]) -> HashMap<String, String> {
    let mut dataset_map = HashMap::new();
    for (relative, fname, path) in dataset.iter() {
        dataset_map.insert(format!("{}+{}", relative, fname), path.to_string());
    }
    dataset_map
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
pub struct Defenses {
    pub description: String,
    pub defenses: Vec<Defense>,
}

pub fn load_defenses(filename: &str) -> Result<Defenses> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // read all lines into a vector
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

    // read the first line as the description
    let description = lines[0].clone();

    // read the rest of the lines as defenses, in parallel, dealing with errors
    let defenses: Vec<Option<Defense>> = lines[1..]
        .par_iter()
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
