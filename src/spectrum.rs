use serde::{Deserialize, Serialize};
use crate::graph::Graph;
use crate::matrix::DenseMatrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>,
}

impl Spectrum {
    pub fn from_graph_laplacian(graph: &Graph) -> Self {
        let l = graph.laplacian();
        let (eigenvalues, eigenvectors) = l.eigendecomposition();
        Self { eigenvalues, eigenvectors }
    }

    pub fn from_graph_normalized_laplacian(graph: &Graph) -> Self {
        let l = graph.normalized_laplacian();
        let (eigenvalues, eigenvectors) = l.eigendecomposition();
        Self { eigenvalues, eigenvectors }
    }

    pub fn from_matrix(matrix: &DenseMatrix) -> Self {
        let (eigenvalues, eigenvectors) = matrix.eigendecomposition();
        Self { eigenvalues, eigenvectors }
    }

    /// Second smallest eigenvalue (= algebraic connectivity)
    pub fn spectral_gap(&self) -> f64 {
        if self.eigenvalues.len() < 2 {
            return 0.0;
        }
        self.eigenvalues[1]
    }

    /// Graph is connected iff λ₁ > 0
    pub fn is_connected(&self) -> bool {
        self.spectral_gap() > 1e-10
    }

    /// Number of connected components = multiplicity of eigenvalue 0
    pub fn number_of_components(&self) -> usize {
        self.eigenvalues.iter().filter(|&&e| e.abs() < 1e-10).count()
    }

    /// Fiedler vector: eigenvector of λ₁ (second smallest eigenvalue)
    pub fn fiedler_vector(&self) -> &[f64] {
        if self.eigenvalues.len() < 2 {
            return &[];
        }
        &self.eigenvectors[1]
    }

    /// Spectral radius: largest |eigenvalue|
    pub fn spectral_radius(&self) -> f64 {
        self.eigenvalues
            .iter()
            .map(|e| e.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Energy: sum of |eigenvalues|
    pub fn energy(&self) -> f64 {
        self.eigenvalues.iter().map(|e| e.abs()).sum()
    }

    /// Spectral entropy: -Σ pᵢ log(pᵢ) where pᵢ = |λᵢ|/Σ|λⱼ|
    pub fn entropy(&self) -> f64 {
        let total: f64 = self.eigenvalues.iter().map(|e| e.abs()).sum();
        if total < 1e-15 {
            return 0.0;
        }
        self.eigenvalues
            .iter()
            .map(|e| {
                let p = e.abs() / total;
                if p > 1e-15 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_connected() {
        let g = Graph::complete(4);
        let s = Spectrum::from_graph_laplacian(&g);
        assert!(s.is_connected());
        assert_eq!(s.number_of_components(), 1);
    }

    #[test]
    fn test_spectrum_disconnected() {
        // Two disconnected triangles
        let mut g = Graph::new(6);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(3, 4, 1.0);
        g.add_edge(4, 5, 1.0);
        g.add_edge(3, 5, 1.0);
        let s = Spectrum::from_graph_laplacian(&g);
        assert!(!s.is_connected());
        assert_eq!(s.number_of_components(), 2);
    }

    #[test]
    fn test_spectral_gap() {
        let g = Graph::path(4);
        let s = Spectrum::from_graph_laplacian(&g);
        // P_4 eigenvalues: 2(1-cos(πk/4)) for k=0,1,2,3
        // k=0: 0, k=1: 2(1-cos(π/4)) = 2-√2 ≈ 0.586
        let expected_gap = 2.0 - 2.0_f64.sqrt();
        assert!((s.spectral_gap() - expected_gap).abs() < 0.01);
    }

    #[test]
    fn test_fiedler_vector_path() {
        let g = Graph::path(4);
        let s = Spectrum::from_graph_laplacian(&g);
        let fv = s.fiedler_vector();
        assert_eq!(fv.len(), 4);
        // Fiedler vector of path should have sign pattern [+, +, -, -] or similar
        // At least one sign change
        let pos = fv.iter().filter(|&&v| v > 1e-10).count();
        let neg = fv.iter().filter(|&&v| v < -1e-10).count();
        assert!(pos > 0 && neg > 0);
    }

    #[test]
    fn test_spectral_entropy() {
        let g = Graph::complete(3);
        let s = Spectrum::from_graph_laplacian(&g);
        // Complete graph K_3: eigenvalues 0, 3, 3
        // Entropy should be finite
        let e = s.entropy();
        assert!(e.is_finite());
        assert!(e >= 0.0);
    }

    #[test]
    fn test_energy() {
        let g = Graph::complete(3);
        let s = Spectrum::from_graph_laplacian(&g);
        // K_3: eigenvalues 0, 3, 3 => energy = 6
        assert!((s.energy() - 6.0).abs() < 0.1);
    }
}
