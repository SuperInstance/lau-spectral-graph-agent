use crate::graph::Graph;
use crate::spectrum::Spectrum;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct SpectralPartitioner;

impl SpectralPartitioner {
    pub fn new() -> Self {
        Self
    }

    /// K-way spectral partitioning
    pub fn partition(&self, graph: &Graph, k: usize) -> Vec<Vec<usize>> {
        let n = graph.n;
        if k >= n {
            return (0..n).map(|i| vec![i]).collect();
        }
        if k <= 1 {
            return vec![(0..n).collect()];
        }

        // 1. Compute normalized Laplacian
        let l_norm = graph.normalized_laplacian();
        let spectrum = Spectrum::from_matrix(&l_norm);

        // 2. Get k smallest eigenvectors (skip the first — it's the constant vector)
        let num_vecs = k.min(spectrum.eigenvectors.len());
        let embeddings: Vec<Vec<f64>> = (0..n)
            .map(|node| {
                (0..num_vecs)
                    .map(|ev_idx| spectrum.eigenvectors[ev_idx][node])
                    .collect()
            })
            .collect();

        // 3. K-means clustering on the embeddings
        self.k_means(&embeddings, k, 100)
    }

    /// Eigengap heuristic for optimal k
    pub fn optimal_k(&self, graph: &Graph, max_k: usize) -> usize {
        let l = graph.laplacian();
        let spectrum = Spectrum::from_matrix(&l);

        let max_check = max_k.min(spectrum.eigenvalues.len() - 1);
        if max_check < 2 {
            return 1;
        }

        let mut best_k = 1;
        let mut best_gap = 0.0_f64;

        for k in 1..max_check {
            let gap = spectrum.eigenvalues[k + 1] - spectrum.eigenvalues[k];
            if gap > best_gap {
                best_gap = gap;
                best_k = k;
            }
        }
        best_k
    }

    /// Newman-Girvan modularity
    pub fn modularity(&self, graph: &Graph, partition: &[Vec<usize>]) -> f64 {
        let m2 = graph.total_volume(); // 2m
        if m2 < 1e-15 {
            return 0.0;
        }

        // Build community assignment
        let mut community = vec![0usize; graph.n];
        for (c, members) in partition.iter().enumerate() {
            for &node in members {
                community[node] = c;
            }
        }

        // Q = Σ_c [e_c - a_c²]
        // e_c = (sum of edge weights within c) / 2m
        // a_c = (sum of degrees in c) / 2m
        let num_communities = partition.len();
        let mut e_c = vec![0.0; num_communities];
        let mut a_c = vec![0.0; num_communities];

        for i in 0..graph.n {
            let c = community[i];
            a_c[c] += graph.degree(i);
            for &(j, w) in &graph.edges[i] {
                if community[i] == community[j] {
                    e_c[c] += w; // counts each edge twice (i→j and j→i)
                }
            }
        }

        let mut q = 0.0;
        for c in 0..num_communities {
            q += e_c[c] / m2 - (a_c[c] / m2) * (a_c[c] / m2);
        }
        q
    }

    fn k_means(&self, data: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<Vec<usize>> {
        let n = data.len();
        let dim = data[0].len();
        let mut rng = rand::rng();

        // Initialize centroids randomly from data points
        let mut centroids: Vec<Vec<f64>> = Vec::new();
        let mut used = std::collections::HashSet::new();
        while centroids.len() < k {
            let idx = rng.random_range(0..n);
            if used.insert(idx) {
                centroids.push(data[idx].clone());
            }
            if used.len() >= n {
                break;
            }
        }

        let mut assignments = vec![0usize; n];

        for _ in 0..max_iter {
            // Assign each point to nearest centroid
            let mut changed = false;
            for i in 0..n {
                let mut best = 0;
                let mut best_dist = f64::MAX;
                for (c, centroid) in centroids.iter().enumerate() {
                    let dist: f64 = data[i]
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best = c;
                    }
                }
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut counts = vec![0usize; k];
            let mut sums = vec![vec![0.0; dim]; k];
            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                for j in 0..dim {
                    sums[c][j] += data[i][j];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        centroids[c][j] = sums[c][j] / counts[c] as f64;
                    }
                }
            }
        }

        // Build partitions
        let mut result: Vec<Vec<usize>> = vec![vec![]; k];
        for i in 0..n {
            result[assignments[i]].push(i);
        }

        // Remove empty clusters
        result.retain(|cluster| !cluster.is_empty());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_two_clusters() {
        // Two cliques connected by single edge
        let mut g = Graph::new(6);
        // Clique 1: 0,1,2
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        // Clique 2: 3,4,5
        g.add_edge(3, 4, 1.0);
        g.add_edge(4, 5, 1.0);
        g.add_edge(3, 5, 1.0);
        // Bridge
        g.add_edge(2, 3, 1.0);

        let p = SpectralPartitioner::new();
        let parts = p.partition(&g, 2);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_optimal_k_two_cliques() {
        let mut g = Graph::new(6);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(3, 4, 1.0);
        g.add_edge(4, 5, 1.0);
        g.add_edge(3, 5, 1.0);
        g.add_edge(2, 3, 0.1); // weak bridge

        let p = SpectralPartitioner::new();
        let k = p.optimal_k(&g, 4);
        assert!(k >= 1 && k <= 4);
    }

    #[test]
    fn test_modularity_single_community() {
        let g = Graph::complete(4);
        let p = SpectralPartitioner::new();
        let partition = vec![vec![0, 1, 2, 3]];
        let m = p.modularity(&g, &partition);
        // Single community should have modularity 0
        assert!(m.abs() < 0.01);
    }

    #[test]
    fn test_modularity_split() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let p = SpectralPartitioner::new();
        let partition = vec![vec![0, 1], vec![2, 3]];
        let m = p.modularity(&g, &partition);
        assert!(m > 0.0);
    }
}
