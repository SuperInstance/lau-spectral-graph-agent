use crate::graph::Graph;
use crate::spectrum::Spectrum;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct CheegerInvariant;

impl CheegerInvariant {
    pub fn new() -> Self {
        Self
    }

    /// Cheeger constant: h(G) = min_S |∂S|/min(|S|, |V\S|)
    /// where ∂S is the edge boundary of S (vertex-based, classical definition)
    pub fn cheeger_constant(&self, graph: &Graph) -> f64 {
        let n = graph.n;
        if n <= 1 {
            return 0.0;
        }

        let max_subset_size = n / 2;

        // Enumerate subsets (only feasible for small n)
        if n <= 20 {
            let mut h = f64::MAX;
            for mask in 1u32..(1u32 << n) {
                let bits = mask.count_ones() as usize;
                if bits > max_subset_size || bits == 0 {
                    continue;
                }
                let set: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
                let boundary: f64 = self.edge_boundary_weight(graph, &set);
                let h_s = boundary / bits.min(n - bits) as f64;
                h = h.min(h_s);
            }
            return h;
        }

        // For larger graphs, use Fiedler vector sweep
        let spectrum = Spectrum::from_graph_laplacian(graph);
        let fv = spectrum.fiedler_vector();
        if fv.is_empty() {
            return 0.0;
        }

        let mut indexed: Vec<(usize, f64)> = fv.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut best_h = f64::MAX;
        for k in 1..n {
            let set: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();
            let boundary = self.edge_boundary_weight(graph, &set);
            let h_s = boundary / k.min(n - k) as f64;
            best_h = best_h.min(h_s);
        }
        best_h
    }

    /// Cheeger inequality bounds from eigenvalue
    pub fn cheeger_from_eigenvalue(&self, lambda: f64) -> (f64, f64) {
        // λ₁/2 ≤ h ≤ √(2λ₁)
        (lambda / 2.0, (2.0 * lambda).sqrt())
    }

    /// Conductance of a set: |∂S|/vol(S)
    pub fn conductance(&self, graph: &Graph, set: &[usize]) -> f64 {
        let vol: f64 = set.iter().map(|&i| graph.degree(i)).sum();
        if vol < 1e-15 {
            return 0.0;
        }
        let boundary = self.edge_boundary_weight(graph, set);
        boundary / vol
    }

    fn edge_boundary_weight(&self, graph: &Graph, set: &[usize]) -> f64 {
        let set_members: HashSet<usize> = set.iter().copied().collect();
        let mut boundary = 0.0;
        for &node in set {
            for &(neighbor, weight) in &graph.edges[node] {
                if !set_members.contains(&neighbor) {
                    boundary += weight;
                }
            }
        }
        boundary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cheeger_constant_complete() {
        let g = Graph::complete(4);
        let c = CheegerInvariant::new();
        let h = c.cheeger_constant(&g);
        // K_4: optimal cut separates 1 vertex: boundary = 3, min(1,3) = 1
        // h = 3/1 = 3.0
        // But 2 vertices: boundary = 4, min(2,2) = 2, h = 4/2 = 2.0
        assert!((h - 2.0).abs() < 0.05, "h = {h}");
    }

    #[test]
    fn test_cheeger_inequality() {
        let g = Graph::path(6);
        let c = CheegerInvariant::new();
        let spectrum = Spectrum::from_graph_laplacian(&g);
        let lambda1 = spectrum.spectral_gap();
        let h = c.cheeger_constant(&g);
        let (lower, upper) = c.cheeger_from_eigenvalue(lambda1);

        // Cheeger inequality: λ₁/2 ≤ h ≤ √(2λ₁)
        assert!(
            h >= lower - 0.01,
            "h ({h}) should be >= λ₁/2 ({lower})"
        );
        assert!(
            h <= upper + 0.01,
            "h ({h}) should be <= √(2λ₁) ({upper})"
        );
    }

    #[test]
    fn test_conductance() {
        let g = Graph::complete(4);
        let c = CheegerInvariant::new();
        let cond = c.conductance(&g, &[0]);
        // Boundary of {0} = edges 0→1, 0→2, 0→3 = 3
        // vol({0}) = 3
        // conductance = 3/3 = 1.0
        assert!((cond - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cheeger_constant_barbell() {
        // Two cliques of size 3 connected by single edge
        let mut g = Graph::new(6);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(3, 4, 1.0);
        g.add_edge(4, 5, 1.0);
        g.add_edge(3, 5, 1.0);
        g.add_edge(2, 3, 1.0); // bridge

        let c = CheegerInvariant::new();
        let h = c.cheeger_constant(&g);
        // Cut {0,1,2} from {3,4,5}: boundary=1, min(3,3) = 3
        // h = 1/3 ≈ 0.333
        assert!((h - 1.0 / 3.0).abs() < 0.05, "h = {h}");
    }
}
