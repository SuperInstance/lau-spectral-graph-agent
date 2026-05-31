use crate::graph::Graph;
use crate::matrix::DenseMatrix;
use crate::spectrum::Spectrum;

#[derive(Debug, Clone)]
pub struct GraphWave;

impl GraphWave {
    pub fn new() -> Self {
        Self
    }

    /// Heat kernel: H_t = e^{-tL}
    pub fn heat_kernel(&self, graph: &Graph, t: f64) -> DenseMatrix {
        let l = graph.laplacian();
        let neg_t_l = l.scale(-t);
        neg_t_l.exp_symmetric()
    }

    /// Spectral wavelet at a specific node and scale
    pub fn wavelet(&self, graph: &Graph, node: usize, scale: f64) -> Vec<f64> {
        let spectrum = Spectrum::from_graph_laplacian(graph);
        let n = graph.n;
        let mut result = vec![0.0; n];

        for k in 0..n {
            let lambda = spectrum.eigenvalues[k];
            let g_lambda = (-scale * lambda).exp(); // heat kernel as scaling function
            for j in 0..n {
                result[j] += g_lambda * spectrum.eigenvectors[k][node] * spectrum.eigenvectors[k][j];
            }
        }
        result
    }

    /// Multi-scale embedding: per-node features at multiple scales
    pub fn multi_scale_embedding(&self, graph: &Graph, scales: &[f64]) -> Vec<Vec<f64>> {
        let spectrum = Spectrum::from_graph_laplacian(graph);
        let n = graph.n;

        // Precompute scaling functions for each eigenvalue at each scale
        let scaled_eigenvecs: Vec<Vec<Vec<f64>>> = scales
            .iter()
            .map(|&t| {
                (0..n)
                    .map(|k| {
                        let g_lambda = (-t * spectrum.eigenvalues[k]).exp();
                        spectrum.eigenvectors[k]
                            .iter()
                            .map(|&v| v * g_lambda)
                            .collect::<Vec<f64>>()
                    })
                    .collect()
            })
            .collect();

        let mut embeddings = vec![vec![0.0; scales.len()]; n];
        for node in 0..n {
            for (s_idx, _) in scales.iter().enumerate() {
                for k in 0..n {
                    embeddings[node][s_idx] +=
                        scaled_eigenvecs[s_idx][k][node] * spectrum.eigenvectors[k][node];
                }
            }
        }
        embeddings
    }

    /// Characteristic function from heat kernel trace
    pub fn characteristic_function(&self, graph: &Graph, t: f64) -> Vec<f64> {
        let h = self.heat_kernel(graph, t);
        // CF = tr(H_t * e^{-itA}) approximation — simplified as diagonal of heat kernel
        (0..graph.n).map(|i| h.data[i][i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_kernel_trace() {
        let g = Graph::path(4);
        let wave = GraphWave::new();
        let spectrum = Spectrum::from_graph_laplacian(&g);
        let t = 1.0;

        // tr(H_t) = Σ e^{-tλᵢ}
        let h = wave.heat_kernel(&g, t);
        let trace = h.trace();

        let expected: f64 = spectrum.eigenvalues.iter().map(|&l| (-t * l).exp()).sum();
        assert!((trace - expected).abs() < 0.1, "trace = {trace}, expected = {expected}");
    }

    #[test]
    fn test_heat_kernel_symmetric() {
        let g = Graph::complete(4);
        let wave = GraphWave::new();
        let h = wave.heat_kernel(&g, 1.0);
        assert!(h.is_symmetric());
    }

    #[test]
    fn test_wavelet_localization() {
        let g = Graph::path(5);
        let wave = GraphWave::new();
        let w = wave.wavelet(&g, 2, 1.0);
        // At small scale, wavelet should be concentrated at node 2
        assert!(w[2] > w[0] || w[2] > w[4], "wavelet should be larger near source");
    }

    #[test]
    fn test_multi_scale_embedding() {
        let g = Graph::complete(4);
        let wave = GraphWave::new();
        let scales = vec![0.1, 1.0, 10.0];
        let emb = wave.multi_scale_embedding(&g, &scales);
        assert_eq!(emb.len(), 4);
        assert_eq!(emb[0].len(), 3);
    }

    #[test]
    fn test_characteristic_function() {
        let g = Graph::complete(3);
        let wave = GraphWave::new();
        let cf = wave.characteristic_function(&g, 1.0);
        assert_eq!(cf.len(), 3);
        // All diagonal entries should be positive
        for &v in &cf {
            assert!(v > 0.0);
        }
    }
}
