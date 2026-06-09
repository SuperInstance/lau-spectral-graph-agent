//! Tutorial: Spectral Graph Theory — Eigenvalues, Cheeger, and Agent Networks
//!
//! Run with: cargo run --example tutorial

use lau_spectral_graph_agent::*;

fn main() {
    println!("=== Lesson 1: Graph Construction ===\n");
    {
        let complete = Graph::complete(5);
        let path = Graph::path(6);
        let cycle = Graph::cycle(6);
        let star = Graph::star(5);

        println!("K5: {} nodes, {} edges", 5, complete.num_edges());
        println!("P6: {} nodes, {} edges", 6, path.num_edges());
        println!("C6: {} nodes, {} edges", 6, cycle.num_edges());
        println!("S5: {} nodes, {} edges", 5, star.num_edges());
        println!("P6 total volume: {:.1}", path.total_volume());
        println!();
    }

    println!("=== Lesson 2: Laplacian Matrices ===\n");
    {
        let g = Graph::cycle(4);
        let adj = g.adjacency_matrix();
        let lap = g.laplacian();
        let norm_lap = g.normalized_laplacian();
        let rw_lap = g.random_walk_laplacian();

        println!("C4 adjacency diagonal sum: {:.0}", (0..4).map(|i| adj.get(i, i)).sum::<f64>());
        println!("Laplacian trace: {:.0}", lap.trace());
        println!("Normalized L trace: {:.4}", norm_lap.trace());
        println!("Random-walk L trace: {:.4}", rw_lap.trace());
        println!();
    }

    println!("=== Lesson 3: Spectral Analysis ===\n");
    {
        let cycle = Graph::cycle(6);
        let spec = Spectrum::from_graph_laplacian(&cycle);

        println!("C6 Laplacian eigenvalues:");
        for (i, &e) in spec.eigenvalues.iter().enumerate() {
            println!("  λ{} = {:.4}", i, e);
        }
        println!("Spectral gap: {:.4}", spec.spectral_gap());
        println!("Connected? {}", spec.is_connected());
        println!("Components: {}", spec.number_of_components());
        println!();

        let path = Graph::path(6);
        let spec_p = Spectrum::from_graph_laplacian(&path);
        println!("P6 spectral gap: {:.4}", spec_p.spectral_gap());
        println!("P6 spectral radius: {:.4}", spec_p.spectral_radius());
        println!();
    }

    println!("=== Lesson 4: Fiedler Vector & Connectivity ===\n");
    {
        let g = Graph::path(8);
        let spec = Spectrum::from_graph_laplacian(&g);

        println!("P8 Fiedler vector (2nd eigenvector):");
        let fv = spec.fiedler_vector();
        for (i, &v) in fv.iter().enumerate() {
            let sign = if v >= 0.0 { "+" } else { "-" };
            println!("  node {}: {:.4} ({})", i, v, sign);
        }
        println!("Sign changes mark the natural graph cut");
        println!();
    }

    println!("=== Lesson 5: Spectral Energy & Entropy ===\n");
    {
        let complete = Graph::complete(6);
        let path = Graph::path(6);

        let spec_k = Spectrum::from_graph_normalized_laplacian(&complete);
        let spec_p = Spectrum::from_graph_normalized_laplacian(&path);

        println!("K6: energy = {:.4}, entropy = {:.4}", spec_k.energy(), spec_k.entropy());
        println!("P6: energy = {:.4}, entropy = {:.4}", spec_p.energy(), spec_p.entropy());
        println!("Complete graphs have more spectral energy (more connectivity)");
        println!();
    }

    println!("=== Lesson 6: Cheeger Inequality ===\n");
    {
        let cheeger = CheegerInvariant::new();

        let cycle = Graph::cycle(8);
        let h_cycle = cheeger.cheeger_constant(&cycle);
        let spec = Spectrum::from_graph_laplacian(&cycle);
        let (h_lo, h_hi) = cheeger.cheeger_from_eigenvalue(spec.spectral_gap());
        println!("C8: h = {:.4}, λ₂ = {:.4}, bounds: [{:.4}, {:.4}]", h_cycle, spec.spectral_gap(), h_lo, h_hi);

        let path = Graph::path(8);
        let h_path = cheeger.cheeger_constant(&path);
        println!("P8: h = {:.4}", h_path);

        // Conductance of a specific set
        let cond = cheeger.conductance(&cycle, &[0, 1, 2]);
        println!("C8 conductance of {{0,1,2}}: {:.4}", cond);
        println!();
    }

    println!("=== Lesson 7: Spectral Partitioning ===\n");
    {
        let partitioner = SpectralPartitioner::new();

        // Create a graph with natural 2-community structure
        let mut g = Graph::new(8);
        // Community 1: 0-3
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(2, 3, 1.0);
        g.add_edge(0, 2, 1.0); g.add_edge(1, 3, 1.0);
        // Community 2: 4-7
        g.add_edge(4, 5, 1.0); g.add_edge(5, 6, 1.0); g.add_edge(6, 7, 1.0);
        g.add_edge(4, 6, 1.0); g.add_edge(5, 7, 1.0);
        // Bridge
        g.add_edge(3, 4, 1.0);

        let parts = partitioner.partition(&g, 2);
        println!("Spectral partition into 2 communities:");
        for (i, p) in parts.iter().enumerate() {
            println!("  Community {}: {:?}", i, p);
        }
        let modularity = partitioner.modularity(&g, &parts);
        println!("Modularity: {:.4}", modularity);

        let opt_k = partitioner.optimal_k(&g, 4);
        println!("Optimal k (max 4): {}", opt_k);
        println!();
    }

    println!("=== Lesson 8: Agent Networks & PageRank ===\n");
    {
        let mut g = Graph::new(5);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0); g.add_edge(3, 4, 1.0);
        g.add_edge(0, 3, 1.0);

        let agents = AgentNetwork::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
            g,
        );

        println!("Agent network:");
        println!("  Algebraic connectivity: {:.4}", agents.algebraic_connectivity());
        println!("  Robustness: {:.4}", agents.robustness());
        let (bn_i, bn_j, bn_w) = agents.bottleneck();
        println!("  Bottleneck: agents {}-{} (weight {:.1})", bn_i, bn_j, bn_w);
        println!("  Bottleneck agents: {:?}", agents.bottleneck_agents());
        println!("  Broadcast order: {:?}", agents.optimal_broadcast_order());

        // PageRank on a different graph
        let mut prg = Graph::new(4);
        prg.add_edge(0, 1, 1.0); prg.add_edge(1, 2, 1.0);
        prg.add_edge(2, 0, 1.0); prg.add_edge(2, 3, 1.0);

        let pr = PageRank::new();
        let ranks = pr.compute(&prg, 0.85, 100);
        println!("\nPageRank scores:");
        for (i, &r) in ranks.iter().enumerate() {
            println!("  Node {}: {:.4}", i, r);
        }

        let trust = pr.trust_rank(&prg, &[0], 0.85);
        println!("\nTrustRank (seed=0):");
        for (i, &t) in trust.iter().enumerate() {
            println!("  Node {}: {:.4}", i, t);
        }
        println!();
    }

    println!("Tutorial complete!");
    println!("Key insight: Eigenvalues encode connectivity, Cheeger bounds it,");
    println!("and spectral methods cut, rank, and compress graphs — all via linear algebra.");
}
