#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::assign_op_pattern, clippy::new_without_default, clippy::collapsible_if)]

pub mod matrix;
pub mod graph;
pub mod spectrum;
pub mod partitioner;
pub mod cheeger;
pub mod wavelets;
pub mod agent_network;
pub mod pagerank;
pub mod sparsifier;

pub use matrix::DenseMatrix;
pub use graph::Graph;
pub use spectrum::Spectrum;
pub use partitioner::SpectralPartitioner;
pub use cheeger::CheegerInvariant;
pub use wavelets::GraphWave;
pub use agent_network::AgentNetwork;
pub use pagerank::PageRank;
pub use sparsifier::GraphSparsifier;
