use std::{fmt, str::FromStr};

use maybenot::Machine;
use serde::{de::Error, ser::SerializeSeq, Deserialize, Serialize, Serializer};
use sha256::digest;

/// A defense consists of zero or more client machines and zero or more server
/// machines. The defense identifier is deterministically derived from the
/// machines in the defense. A note field is provided for additional
/// information.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Defense {
    #[serde(
        serialize_with = "serialize_machines",
        deserialize_with = "deserialize_machines"
    )]
    pub client: Vec<Machine>,
    #[serde(
        serialize_with = "serialize_machines",
        deserialize_with = "deserialize_machines"
    )]
    pub server: Vec<Machine>,
    pub note: Option<String>,
    #[serde(skip)]
    id: String,
}

impl Defense {
    pub fn new(client: Vec<Machine>, server: Vec<Machine>) -> Self {
        let id = get_id(&client, &server);
        Self {
            client,
            server,
            id,
            note: None,
        }
    }

    pub fn num_client_machines(&self) -> usize {
        self.client.len()
    }

    pub fn num_server_machines(&self) -> usize {
        self.server.len()
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    // update the ID of the defense: this is necessary if the machines in the
    // defense have changed
    pub fn update_id(&mut self) {
        self.id = get_id(&self.client, &self.server);
    }
}

// custom serialization function for machines using Maybenot format
fn serialize_machines<S>(machines: &Vec<Machine>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(machines.len()))?;
    for m in machines {
        seq.serialize_element(&m.serialize())?;
    }

    seq.end()
}

// custom deserialization function for machines using Maybenot format
fn deserialize_machines<'de, D>(deserializer: D) -> Result<Vec<Machine>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let machines: Vec<String> = serde::Deserialize::deserialize(deserializer)?;
    let mut result = Vec::new();
    for m in machines {
        match Machine::from_str(&m) {
            Ok(machine) => result.push(machine),
            Err(e) => return Err(Error::custom(format!("invalid machine format: {e}"))),
        }
    }
    Ok(result)
}

fn get_id(client: &[Machine], server: &[Machine]) -> String {
    // allocate id with capacity 32 bytes for each machine
    let mut id = String::with_capacity(client.len() * 32 + server.len() * 32);
    for m in client {
        id.push_str(&m.name());
    }
    for m in server {
        id.push_str(&m.name());
    }
    let s = digest(id);
    s[0..32].to_string()
}

impl fmt::Display for Defense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // id and note
        writeln!(f, "id: {}", self.id)?;
        if let Some(note) = &self.note {
            writeln!(f, "note: {note}")?;
        }
        writeln!(f, "client machine(s):")?;
        for m in &self.client {
            writeln!(f, "{}", m.serialize())?;
        }
        writeln!(f, "server machine(s):")?;
        for m in &self.server {
            writeln!(f, "{}", m.serialize())?;
        }
        Ok(())
    }
}

impl PartialEq for Defense {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Defense {}

impl PartialOrd for Defense {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Defense {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
