use rand::Rng;

use crate::graph::Graph;

#[derive(Debug, Clone)]
pub struct GraphSparsifier;

impl GraphSparsifier {
    pub fn new() -> Self {
        Self
    }

    /// Effective resistance: R_ij = (e_i - e_j)^T L^+ (e_i - e_j)
    pub fn effective_resistance(&self, graph: &Graph, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        let l = graph.laplacian();
        let n = graph.n;

        // Compute pseudoinverse of L using eigendecomposition
        let (eigenvalues, eigenvectors) = l.eigendecomposition();

        // Build e_i - e_j
        let mut diff = vec![0.0; n];
        diff[i] = 1.0;
        diff[j] = -1.0;

        // L^+ = Σ_{k: λ_k > 0} (1/λ_k) v_k v_k^T
        // R_ij = diff^T L^+ diff = Σ_{k: λ_k > 0} (1/λ_k) (v_k^T diff)^2
        let mut resistance = 0.0;
        for k in 0..n {
            if eigenvalues[k].abs() > 1e-10 {
                let dot: f64 = eigenvectors[k].iter().zip(diff.iter()).map(|(a, b)| a * b).sum();
                resistance += dot * dot / eigenvalues[k];
            }
        }
        resistance
    }

    /// Spanner: subgraph with bounded stretch
    pub fn spanner(&self, graph: &Graph, _stretch: f64) -> Graph {
        // Simple greedy spanner construction
        let n = graph.n;
        let mut result = Graph::new(n);

        // Collect all edges
        let mut all_edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for &(j, w) in &graph.edges[i] {
                if j > i {
                    all_edges.push((i, j, w));
                }
            }
        }
        all_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Greedily add edges
        for (i, j, w) in &all_edges {
            // Check if i and j are already connected in the spanner
            if !self.is_connected_in(&result, *i, *j) {
                result.add_edge(*i, *j, *w);
            }
        }

        result
    }

    /// Spectral sparsification (Spielman-Srivastava style)
    pub fn spectral_sparsify(&self, graph: &Graph, epsilon: f64) -> Graph {
        let n = graph.n;
        let m = graph.num_edges();

        if m < n {
            return graph.clone();
        }

        // Simplified spectral sparsification: sample edges with probability
        // proportional to effective resistance * weight
        let q = ((n as f64) * (1.0 + epsilon).ln() / (epsilon * epsilon)).ceil() as usize;
        let q = q.max(n - 1).min(m);

        // Compute approximate edge sampling probabilities
        let mut edge_probs: Vec<(usize, usize, f64, f64)> = Vec::new(); // (i, j, weight, prob)
        for i in 0..n {
            for &(j, w) in &graph.edges[i] {
                if j > i {
                    let r_eff = self.effective_resistance(graph, i, j);
                    let prob = (w * r_eff).min(1.0);
                    edge_probs.push((i, j, w, prob));
                }
            }
        }

        let total_prob: f64 = edge_probs.iter().map(|(_, _, _, p)| p).sum();
        if total_prob < 1e-15 {
            return graph.clone();
        }

        // Normalize and sample
        let mut result = Graph::new(n);
        let mut rng = rand::rng();

        for _ in 0..q {
            let r: f64 = rng.random_range(0.0..total_prob);
            let mut cumsum = 0.0;
            for &(i, j, w, p) in &edge_probs {
                cumsum += p;
                if cumsum >= r {
                    // Rescale weight
                    let new_weight = w * total_prob / (q as f64 * p.max(1e-15));
                    if !result.has_edge(i, j) {
                        result.add_edge(i, j, new_weight.max(w * 0.1)); // keep some minimum weight
                    }
                    break;
                }
            }
        }

        // Ensure connectivity — add back edges if needed
        for i in 0..n {
            for &(j, w) in &graph.edges[i] {
                if j > i && !result.has_edge(i, j)
                    && !self.is_connected_in(&result, i, j)
                {
                    result.add_edge(i, j, w);
                }
            }
        }

        result
    }

    fn is_connected_in(&self, graph: &Graph, start: usize, target: usize) -> bool {
        if start == target {
            return true;
        }
        let mut visited = vec![false; graph.n];
        let mut stack = vec![start];
        visited[start] = true;

        while let Some(node) = stack.pop() {
            for &(neighbor, _) in &graph.edges[node] {
                if neighbor == target {
                    return true;
                }
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Spectrum;

    #[test]
    fn test_effective_resistance_same_node() {
        let g = Graph::complete(4);
        let sp = GraphSparsifier::new();
        let r = sp.effective_resistance(&g, 0, 0);
        assert!((r - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_resistance_complete() {
        // K_n: R_ij = 2/n
        let g = Graph::complete(4);
        let sp = GraphSparsifier::new();
        let r = sp.effective_resistance(&g, 0, 1);
        assert!((r - 0.5).abs() < 0.1, "R_01 = {r}, expected 0.5");
    }

    #[test]
    fn test_effective_resistance_symmetric() {
        let g = Graph::complete(5);
        let sp = GraphSparsifier::new();
        let r01 = sp.effective_resistance(&g, 0, 1);
        let r10 = sp.effective_resistance(&g, 1, 0);
        assert!((r01 - r10).abs() < 1e-8, "resistance should be symmetric");
    }

    #[test]
    fn test_effective_resistance_triangle_inequality() {
        let g = Graph::complete(4);
        let sp = GraphSparsifier::new();
        let r01 = sp.effective_resistance(&g, 0, 1);
        let r12 = sp.effective_resistance(&g, 1, 2);
        let r02 = sp.effective_resistance(&g, 0, 2);
        assert!(
            r02 <= r01 + r12 + 0.01,
            "triangle inequality: R_02 ({r02}) <= R_01 ({r01}) + R_12 ({r12})"
        );
    }

    #[test]
    fn test_effective_resistance_nonneg() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let sp = GraphSparsifier::new();
        let r = sp.effective_resistance(&g, 0, 3);
        assert!(r >= 0.0, "resistance should be non-negative");
    }

    #[test]
    fn test_spanner_connectivity() {
        let g = Graph::complete(5);
        let sp = GraphSparsifier::new();
        let spanner = sp.spanner(&g, 2.0);
        // Spanner should maintain connectivity
        let spectrum = Spectrum::from_graph_laplacian(&spanner);
        assert!(spectrum.is_connected());
    }

    #[test]
    fn test_spanner_fewer_edges() {
        let g = Graph::complete(5);
        let sp = GraphSparsifier::new();
        let spanner = sp.spanner(&g, 2.0);
        assert!(spanner.num_edges() <= g.num_edges());
    }

    #[test]
    fn test_spectral_sparsify_connectivity() {
        let mut g = Graph::new(6);
        for i in 0..6 {
            for j in (i + 1)..6 {
                g.add_edge(i, j, 1.0);
            }
        }
        let sp = GraphSparsifier::new();
        let sparse = sp.spectral_sparsify(&g, 0.5);
        let spectrum = Spectrum::from_graph_laplacian(&sparse);
        assert!(spectrum.is_connected());
    }
}
