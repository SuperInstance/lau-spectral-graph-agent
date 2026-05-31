use serde::{Deserialize, Serialize};
use crate::matrix::DenseMatrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<Vec<(usize, f64)>>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: vec![vec![]; n],
        }
    }

    pub fn add_edge(&mut self, i: usize, j: usize, weight: f64) {
        self.edges[i].push((j, weight));
        self.edges[j].push((i, weight));
    }

    pub fn adjacency_matrix(&self) -> DenseMatrix {
        let mut m = DenseMatrix::zeros(self.n, self.n);
        for (i, neighbors) in self.edges.iter().enumerate() {
            for &(j, w) in neighbors {
                m.data[i][j] = w;
            }
        }
        m
    }

    pub fn degree_matrix(&self) -> DenseMatrix {
        let mut m = DenseMatrix::zeros(self.n, self.n);
        for (i, neighbors) in self.edges.iter().enumerate() {
            let deg: f64 = neighbors.iter().map(|(_, w)| w).sum();
            m.data[i][i] = deg;
        }
        m
    }

    pub fn laplacian(&self) -> DenseMatrix {
        self.degree_matrix().sub(&self.adjacency_matrix())
    }

    pub fn normalized_laplacian(&self) -> DenseMatrix {
        let d = self.degree_matrix();
        let l = self.laplacian();
        // L_sym = D^{-1/2} L D^{-1/2}
        let mut d_inv_sqrt = vec![0.0; self.n];
        for i in 0..self.n {
            if d.data[i][i] > 1e-15 {
                d_inv_sqrt[i] = 1.0 / d.data[i][i].sqrt();
            }
        }
        let mut result = DenseMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                result.data[i][j] = d_inv_sqrt[i] * l.data[i][j] * d_inv_sqrt[j];
            }
        }
        result
    }

    pub fn random_walk_laplacian(&self) -> DenseMatrix {
        let d = self.degree_matrix();
        let l = self.laplacian();
        // L_rw = D^{-1} L
        let mut d_inv = vec![0.0; self.n];
        for i in 0..self.n {
            if d.data[i][i] > 1e-15 {
                d_inv[i] = 1.0 / d.data[i][i];
            }
        }
        let mut result = DenseMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                result.data[i][j] = d_inv[i] * l.data[i][j];
            }
        }
        result
    }

    pub fn degree(&self, i: usize) -> f64 {
        self.edges[i].iter().map(|(_, w)| w).sum()
    }

    pub fn total_volume(&self) -> f64 {
        (0..self.n).map(|i| self.degree(i)).sum()
    }

    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        self.edges[i].iter().any(|&(neighbor, _)| neighbor == j)
    }

    pub fn edge_weight(&self, i: usize, j: usize) -> f64 {
        self.edges[i]
            .iter()
            .find(|&&(neighbor, _)| neighbor == j)
            .map(|&(_, w)| w)
            .unwrap_or(0.0)
    }

    pub fn num_edges(&self) -> usize {
        self.edges.iter().map(|e| e.len()).sum::<usize>() / 2
    }

    /// Complete graph K_n
    pub fn complete(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j, 1.0);
            }
        }
        g
    }

    /// Path graph P_n
    pub fn path(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1, 1.0);
        }
        g
    }

    /// Cycle graph C_n
    pub fn cycle(n: usize) -> Self {
        let mut g = Self::path(n);
        if n > 2 {
            g.add_edge(0, n - 1, 1.0);
        }
        g
    }

    /// Star graph S_n (1 center + n-1 leaves)
    pub fn star(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 1..n {
            g.add_edge(0, i, 1.0);
        }
        g
    }

    /// Create subgraph induced by a subset of vertices (renumbered to 0..k)
    pub fn induced_subgraph(&self, vertices: &[usize]) -> Self {
        let n = vertices.len();
        let mut index_map = vec![0usize; self.n];
        for (new_idx, &old_idx) in vertices.iter().enumerate() {
            index_map[old_idx] = new_idx;
        }
        let mut g = Self::new(n);
        for (new_i, &old_i) in vertices.iter().enumerate() {
            for &(old_j, w) in &self.edges[old_i] {
                if vertices.contains(&old_j) && new_i < index_map[old_j] {
                    g.add_edge(new_i, index_map[old_j], w);
                }
            }
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_complete() {
        let g = Graph::complete(4);
        let l = g.laplacian();
        // K_4: diagonal = 3, off-diagonal = -1
        assert!((l.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((l.get(0, 1) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_path() {
        let g = Graph::path(3);
        let l = g.laplacian();
        // P_3: [[1,-1,0],[-1,2,-1],[0,-1,1]]
        assert!((l.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((l.get(1, 1) - 2.0).abs() < 1e-10);
        assert!((l.get(0, 2)).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_laplacian() {
        let g = Graph::complete(3);
        let l_norm = g.normalized_laplacian();
        assert!(l_norm.is_symmetric());
    }

    #[test]
    fn test_graph_serde() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 2.5);
        g.add_edge(1, 2, 1.0);
        let json = serde_json::to_string(&g).unwrap();
        let g2: Graph = serde_json::from_str(&json).unwrap();
        assert_eq!(g2.n, 3);
        assert_eq!(g2.edges[0].len(), 1);
    }

    #[test]
    fn test_star_graph() {
        let g = Graph::star(5);
        assert_eq!(g.edges[0].len(), 4); // center has degree 4
        for i in 1..5 {
            assert_eq!(g.edges[i].len(), 1); // leaves have degree 1
        }
    }
}
