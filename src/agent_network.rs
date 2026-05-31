use serde::{Deserialize, Serialize};
use crate::graph::Graph;
use crate::spectrum::Spectrum;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNetwork {
    pub agents: Vec<String>,
    pub graph: Graph,
}

impl AgentNetwork {
    pub fn new(agents: Vec<String>, graph: Graph) -> Self {
        assert_eq!(agents.len(), graph.n);
        Self { agents, graph }
    }

    /// Algebraic connectivity: how well-connected is the network
    pub fn algebraic_connectivity(&self) -> f64 {
        let spectrum = Spectrum::from_graph_laplacian(&self.graph);
        spectrum.spectral_gap()
    }

    /// The weakest link (by Fiedler vector — the edge crossing the spectral cut)
    pub fn bottleneck(&self) -> (usize, usize, f64) {
        let spectrum = Spectrum::from_graph_laplacian(&self.graph);
        let fv = spectrum.fiedler_vector();

        let mut best_edge = (0, 1);
        let mut best_score = 0.0_f64;

        for i in 0..self.graph.n {
            for &(j, _w) in &self.graph.edges[i] {
                if j > i {
                    // Edges crossing the Fiedler cut have different signs
                    let score = (fv[i] - fv[j]).abs() / (fv[i].abs() + fv[j].abs() + 1e-15);
                    if score > best_score {
                        best_score = score;
                        best_edge = (i, j);
                    }
                }
            }
        }

        (best_edge.0, best_edge.1, best_score)
    }

    /// Robustness: algebraic connectivity after removing the best-connected node
    pub fn robustness(&self) -> f64 {
        let centrality = self.eigenvector_centrality();
        let best_node = centrality
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let remaining: Vec<usize> = (0..self.graph.n).filter(|&i| i != best_node).collect();
        let subgraph = self.graph.induced_subgraph(&remaining);
        if subgraph.n == 0 {
            return 0.0;
        }
        let spectrum = Spectrum::from_graph_laplacian(&subgraph);
        spectrum.spectral_gap()
    }

    /// Agents on the spectral cut boundary
    pub fn bottleneck_agents(&self) -> Vec<usize> {
        let spectrum = Spectrum::from_graph_laplacian(&self.graph);
        let fv = spectrum.fiedler_vector();
        let median = {
            let mut sorted: Vec<f64> = fv.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = sorted.len() / 2;
            sorted[mid]
        };

        // Nodes near the Fiedler value that changes sign
        let mut boundary = vec![];
        for i in 0..self.graph.n {
            if (fv[i] - median).abs() < 0.3 * (fv.iter().cloned().fold(f64::MAX, f64::min).abs()
                + fv.iter().cloned().fold(f64::MIN, f64::max).abs())
            {
                boundary.push(i);
            }
        }

        if boundary.is_empty() {
            // Fallback: nodes with edges crossing the cut
            let pos: std::collections::HashSet<usize> = fv
                .iter()
                .enumerate()
                .filter(|(_, &v)| v >= 0.0)
                .map(|(i, _)| i)
                .collect();
            for i in 0..self.graph.n {
                for &(j, _) in &self.graph.edges[i] {
                    if pos.contains(&i) != pos.contains(&j) {
                        boundary.push(i);
                        break;
                    }
                }
            }
        }

        boundary
    }

    /// Sort by eigenvector centrality for broadcast ordering
    pub fn optimal_broadcast_order(&self) -> Vec<usize> {
        let centrality = self.eigenvector_centrality();
        let mut indexed: Vec<(usize, f64)> = centrality.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().map(|(i, _)| i).collect()
    }

    fn eigenvector_centrality(&self) -> Vec<f64> {
        let a = self.graph.adjacency_matrix();
        let (_eigenvalues, eigenvectors) = a.eigendecomposition();
        // Eigenvector corresponding to largest eigenvalue
        if eigenvectors.is_empty() {
            return vec![1.0; self.graph.n];
        }
        let last = eigenvectors.last().unwrap();
        let norm: f64 = last.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return vec![1.0; self.graph.n];
        }
        last.iter().map(|&v| v.abs() / norm).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_network() -> AgentNetwork {
        let agents = vec![
            "alpha".into(),
            "beta".into(),
            "gamma".into(),
            "delta".into(),
        ];
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(0, 3, 1.0);
        AgentNetwork::new(agents, g)
    }

    #[test]
    fn test_algebraic_connectivity() {
        let net = make_test_network();
        let ac = net.algebraic_connectivity();
        assert!(ac > 0.0, "connected graph should have positive algebraic connectivity");
    }

    #[test]
    fn test_bottleneck() {
        let net = make_test_network();
        let (i, j, score) = net.bottleneck();
        assert!(i < 4 && j < 4);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_robustness() {
        let net = make_test_network();
        let r = net.robustness();
        assert!(r >= 0.0);
    }

    #[test]
    fn test_bottleneck_agents() {
        let net = make_test_network();
        let agents = net.bottleneck_agents();
        assert!(!agents.is_empty());
    }

    #[test]
    fn test_broadcast_order() {
        let net = make_test_network();
        let order = net.optimal_broadcast_order();
        assert_eq!(order.len(), 4);
        // Should be a permutation of 0..4
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_agent_network_serde() {
        let net = make_test_network();
        let json = serde_json::to_string(&net).unwrap();
        let net2: AgentNetwork = serde_json::from_str(&json).unwrap();
        assert_eq!(net2.agents.len(), 4);
    }
}
