pub(crate) mod decay;
pub(crate) mod config;
pub(crate) mod projection;
pub(crate) mod analysis;
pub(crate) mod filter;
pub(crate) mod persistence;

pub use config::FFBFConfig;
pub use decay::{DecayMode, TickShape};
pub use filter::FFBF;
pub use persistence::{from_json, load, save, to_json};
