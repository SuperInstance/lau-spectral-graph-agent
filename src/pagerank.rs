use crate::graph::Graph;

#[derive(Debug, Clone)]
pub struct PageRank;

impl PageRank {
    pub fn new() -> Self {
        Self
    }

    /// Standard PageRank via power iteration
    pub fn compute(&self, graph: &Graph, damping: f64, iterations: usize) -> Vec<f64> {
        let n = graph.n;
        if n == 0 {
            return vec![];
        }
        let personalization = vec![1.0 / n as f64; n];
        self.power_iteration(graph, damping, iterations, &personalization)
    }

    /// Personalized PageRank
    pub fn personalized(
        &self,
        graph: &Graph,
        personalization: &[f64],
        damping: f64,
    ) -> Vec<f64> {
        let n = graph.n;
        if n == 0 {
            return vec![];
        }
        self.power_iteration(graph, damping, 100, personalization)
    }

    /// TrustRank: personalized PageRank with trusted seed nodes
    pub fn trust_rank(&self, graph: &Graph, trusted: &[usize], damping: f64) -> Vec<f64> {
        let n = graph.n;
        if n == 0 {
            return vec![];
        }
        let mut personalization = vec![0.0; n];
        if trusted.is_empty() {
            return vec![1.0 / n as f64; n];
        }
        for &node in trusted {
            personalization[node] = 1.0 / trusted.len() as f64;
        }
        self.power_iteration(graph, damping, 100, &personalization)
    }

    fn power_iteration(
        &self,
        graph: &Graph,
        damping: f64,
        iterations: usize,
        personalization: &[f64],
    ) -> Vec<f64> {
        let n = graph.n;
        let mut ranks = vec![1.0 / n as f64; n];

        // Build transition matrix from adjacency
        // For undirected: transition probability i→j = w_ij / deg(i)
        let degrees: Vec<f64> = (0..n).map(|i| graph.degree(i)).collect();

        for _ in 0..iterations {
            let mut new_ranks = vec![0.0; n];

            for i in 0..n {
                if degrees[i] < 1e-15 {
                    // Dangling node: distribute equally
                    for j in 0..n {
                        new_ranks[j] += ranks[i] / n as f64;
                    }
                } else {
                    for &(j, w) in &graph.edges[i] {
                        new_ranks[j] += ranks[i] * w / degrees[i];
                    }
                }
            }

            for j in 0..n {
                ranks[j] = (1.0 - damping) * personalization[j] + damping * new_ranks[j];
            }

            // Normalize
            let sum: f64 = ranks.iter().sum();
            if sum > 1e-15 {
                for r in &mut ranks {
                    *r /= sum;
                }
            }
        }

        ranks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_sums_to_one() {
        let g = Graph::complete(5);
        let pr = PageRank::new();
        let ranks = pr.compute(&g, 0.85, 100);
        let sum: f64 = ranks.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "PageRank sums to {sum}");
    }

    #[test]
    fn test_pagerank_star_center_important() {
        let g = Graph::star(5);
        let pr = PageRank::new();
        let ranks = pr.compute(&g, 0.85, 200);
        // Center should have highest PageRank
        let center_rank = ranks[0];
        for i in 1..5 {
            assert!(
                center_rank > ranks[i],
                "center ({center_rank}) should be > leaf {i} ({})",
                ranks[i]
            );
        }
    }

    #[test]
    fn test_pagerank_path_middle() {
        let g = Graph::path(5);
        let pr = PageRank::new();
        let ranks = pr.compute(&g, 0.85, 200);
        // In a path, middle nodes have higher degree → higher PageRank
        assert!(ranks[2] > ranks[0] || (ranks[2] - ranks[0]).abs() < 0.01);
    }

    #[test]
    fn test_personalized_pagerank() {
        let g = Graph::complete(4);
        let pr = PageRank::new();
        let pers = vec![0.5, 0.5, 0.0, 0.0];
        let ranks = pr.personalized(&g, &pers, 0.85);
        let sum: f64 = ranks.iter().sum();
        assert!((sum - 1.0).abs() < 0.1, "PPR sums to {sum}");
    }

    #[test]
    fn test_trust_rank() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let pr = PageRank::new();
        let ranks = pr.trust_rank(&g, &[0], 0.85);
        assert_eq!(ranks.len(), 4);
        // Trusted node should have high rank
        assert!(ranks[0] > 0.0);
    }

    #[test]
    fn test_pagerank_all_positive() {
        let g = Graph::complete(5);
        let pr = PageRank::new();
        let ranks = pr.compute(&g, 0.85, 100);
        for &r in &ranks {
            assert!(r > 0.0, "All ranks should be positive");
        }
    }
}
