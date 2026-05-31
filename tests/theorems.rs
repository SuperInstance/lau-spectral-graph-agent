// Integration tests: theorem verification
use lau_spectral_graph_agent::*;

// ============ Theorem 1: Multiplicity of eigenvalue 0 = number of connected components ============
#[test]
fn test_theorem1_single_component() {
    let g = Graph::complete(5);
    let s = Spectrum::from_graph_laplacian(&g);
    assert_eq!(s.number_of_components(), 1);
}

#[test]
fn test_theorem1_two_components() {
    let mut g = Graph::new(6);
    // Component 1: 0-1-2
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    // Component 2: 3-4-5
    g.add_edge(3, 4, 1.0);
    g.add_edge(4, 5, 1.0);
    let s = Spectrum::from_graph_laplacian(&g);
    assert_eq!(s.number_of_components(), 2);
}

#[test]
fn test_theorem1_three_components() {
    let mut g = Graph::new(6);
    g.add_edge(0, 1, 1.0);
    g.add_edge(2, 3, 1.0);
    g.add_edge(4, 5, 1.0);
    let s = Spectrum::from_graph_laplacian(&g);
    assert_eq!(s.number_of_components(), 3);
}

#[test]
fn test_theorem1_isolated_nodes() {
    let mut g = Graph::new(4);
    g.add_edge(0, 1, 1.0);
    // nodes 2, 3 are isolated
    let s = Spectrum::from_graph_laplacian(&g);
    assert_eq!(s.number_of_components(), 3);
}

// ============ Theorem 2: Cheeger inequality ============
#[test]
fn test_theorem2_cheeger_path() {
    let g = Graph::path(8);
    let c = CheegerInvariant::new();
    let spectrum = Spectrum::from_graph_laplacian(&g);
    let lambda1 = spectrum.spectral_gap();
    let h = c.cheeger_constant(&g);
    let (lower, upper) = c.cheeger_from_eigenvalue(lambda1);
    assert!(h >= lower - 0.01, "h={h}, lower={lower}");
    assert!(h <= upper + 0.01, "h={h}, upper={upper}");
}

#[test]
fn test_theorem2_cheeger_cycle() {
    let g = Graph::cycle(6);
    let c = CheegerInvariant::new();
    let spectrum = Spectrum::from_graph_laplacian(&g);
    let lambda1 = spectrum.spectral_gap();
    let h = c.cheeger_constant(&g);
    let (lower, upper) = c.cheeger_from_eigenvalue(lambda1);
    assert!(h >= lower - 0.01);
    assert!(h <= upper + 0.01);
}

#[test]
fn test_theorem2_cheeger_star() {
    let g = Graph::star(5);
    let c = CheegerInvariant::new();
    let spectrum = Spectrum::from_graph_laplacian(&g);
    let lambda1 = spectrum.spectral_gap();
    let h = c.cheeger_constant(&g);
    let (lower, upper) = c.cheeger_from_eigenvalue(lambda1);
    assert!(h >= lower - 0.01);
    assert!(h <= upper + 0.01);
}

// ============ Theorem 3: Complete graph eigenvalues ============
#[test]
fn test_theorem3_complete_k4() {
    let g = Graph::complete(4);
    let s = Spectrum::from_graph_laplacian(&g);
    // K_4: eigenvalues are 0, 4, 4, 4
    assert!(s.eigenvalues[0].abs() < 0.1, "λ₀ = {}", s.eigenvalues[0]);
    for k in 1..4 {
        assert!((s.eigenvalues[k] - 4.0).abs() < 0.1, "λ_{k} = {}", s.eigenvalues[k]);
    }
}

#[test]
fn test_theorem3_complete_k3() {
    let g = Graph::complete(3);
    let s = Spectrum::from_graph_laplacian(&g);
    // K_3: eigenvalues are 0, 3, 3
    assert!(s.eigenvalues[0].abs() < 0.1);
    assert!((s.eigenvalues[1] - 3.0).abs() < 0.1);
    assert!((s.eigenvalues[2] - 3.0).abs() < 0.1);
}

#[test]
fn test_theorem3_complete_k5() {
    let g = Graph::complete(5);
    let s = Spectrum::from_graph_laplacian(&g);
    // K_5: eigenvalues are 0, 5, 5, 5, 5
    assert!(s.eigenvalues[0].abs() < 0.1);
    for k in 1..5 {
        assert!((s.eigenvalues[k] - 5.0).abs() < 0.2, "λ_{k} = {}", s.eigenvalues[k]);
    }
}

// ============ Theorem 4: Path graph eigenvalues ============
#[test]
fn test_theorem4_path_p4() {
    let g = Graph::path(4);
    let s = Spectrum::from_graph_laplacian(&g);
    // Eigenvalues: 2(1-cos(πk/4)), k=0,1,2,3
    let expected: Vec<f64> = (0..4).map(|k| 2.0 * (1.0 - (std::f64::consts::PI * k as f64 / 4.0).cos())).collect();
    for i in 0..4 {
        assert!(
            (s.eigenvalues[i] - expected[i]).abs() < 0.05,
            "λ[{i}] = {}, expected {}", s.eigenvalues[i], expected[i]
        );
    }
}

#[test]
fn test_theorem4_path_p5() {
    let g = Graph::path(5);
    let s = Spectrum::from_graph_laplacian(&g);
    let expected: Vec<f64> = (0..5).map(|k| 2.0 * (1.0 - (std::f64::consts::PI * k as f64 / 5.0).cos())).collect();
    for i in 0..5 {
        assert!(
            (s.eigenvalues[i] - expected[i]).abs() < 0.05,
            "λ[{i}] = {}, expected {}", s.eigenvalues[i], expected[i]
        );
    }
}

// ============ Theorem 5: Cycle graph eigenvalues ============
#[test]
fn test_theorem5_cycle_c4() {
    let g = Graph::cycle(4);
    let s = Spectrum::from_graph_laplacian(&g);
    // Eigenvalues: 2(1-cos(2πk/4)), k=0,1,2,3
    let expected: Vec<f64> = (0..4)
        .map(|k| 2.0 * (1.0 - (2.0 * std::f64::consts::PI * k as f64 / 4.0).cos()))
        .collect();
    let mut exp_sorted = expected;
    exp_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..4 {
        assert!(
            (s.eigenvalues[i] - exp_sorted[i]).abs() < 0.05,
            "λ[{i}] = {}, expected {}", s.eigenvalues[i], exp_sorted[i]
        );
    }
}

#[test]
fn test_theorem5_cycle_c6() {
    let g = Graph::cycle(6);
    let s = Spectrum::from_graph_laplacian(&g);
    let expected: Vec<f64> = (0..6)
        .map(|k| 2.0 * (1.0 - (2.0 * std::f64::consts::PI * k as f64 / 6.0).cos()))
        .collect();
    let mut exp_sorted = expected;
    exp_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..6 {
        assert!(
            (s.eigenvalues[i] - exp_sorted[i]).abs() < 0.05,
            "λ[{i}] = {}, expected {}", s.eigenvalues[i], exp_sorted[i]
        );
    }
}

// ============ Theorem 6: Star graph λ₁ = 1 ============
#[test]
fn test_theorem6_star_s4() {
    let g = Graph::star(4);
    let s = Spectrum::from_graph_laplacian(&g);
    let lambda1 = s.spectral_gap();
    assert!((lambda1 - 1.0).abs() < 0.1, "λ₁ = {lambda1}, expected 1.0");
}

#[test]
fn test_theorem6_star_s6() {
    let g = Graph::star(6);
    let s = Spectrum::from_graph_laplacian(&g);
    let lambda1 = s.spectral_gap();
    assert!((lambda1 - 1.0).abs() < 0.1, "λ₁ = {lambda1}, expected 1.0");
}

#[test]
fn test_theorem6_star_s8() {
    let g = Graph::star(8);
    let s = Spectrum::from_graph_laplacian(&g);
    let lambda1 = s.spectral_gap();
    assert!((lambda1 - 1.0).abs() < 0.15, "λ₁ = {lambda1}, expected 1.0");
}

// ============ Theorem 7: PageRank sums to 1 ============
#[test]
fn test_theorem7_pagerank_sums_path() {
    let g = Graph::path(5);
    let pr = PageRank::new();
    let ranks = pr.compute(&g, 0.85, 200);
    let sum: f64 = ranks.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
}

#[test]
fn test_theorem7_pagerank_sums_cycle() {
    let g = Graph::cycle(6);
    let pr = PageRank::new();
    let ranks = pr.compute(&g, 0.85, 200);
    let sum: f64 = ranks.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
}

#[test]
fn test_theorem7_pagerank_sums_star() {
    let g = Graph::star(5);
    let pr = PageRank::new();
    let ranks = pr.compute(&g, 0.85, 200);
    let sum: f64 = ranks.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
}

// ============ Theorem 8: Heat kernel trace ============
#[test]
fn test_theorem8_heat_trace_path() {
    let g = Graph::path(5);
    let wave = GraphWave::new();
    let spectrum = Spectrum::from_graph_laplacian(&g);
    let t = 0.5;
    let h = wave.heat_kernel(&g, t);
    let trace = h.trace();
    let expected: f64 = spectrum.eigenvalues.iter().map(|&l| (-t * l).exp()).sum();
    assert!((trace - expected).abs() < 0.1, "trace={trace}, expected={expected}");
}

#[test]
fn test_theorem8_heat_trace_cycle() {
    let g = Graph::cycle(4);
    let wave = GraphWave::new();
    let spectrum = Spectrum::from_graph_laplacian(&g);
    let t = 1.0;
    let h = wave.heat_kernel(&g, t);
    let trace = h.trace();
    let expected: f64 = spectrum.eigenvalues.iter().map(|&l| (-t * l).exp()).sum();
    assert!((trace - expected).abs() < 0.1);
}

// ============ Theorem 9: Spectral partitioning finds good cut ============
#[test]
fn test_theorem9_spectral_cut_barbell() {
    // Two cliques with a weak bridge
    let mut g = Graph::new(6);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(0, 2, 1.0);
    g.add_edge(3, 4, 1.0);
    g.add_edge(4, 5, 1.0);
    g.add_edge(3, 5, 1.0);
    g.add_edge(2, 3, 1.0);

    let p = SpectralPartitioner::new();
    let parts = p.partition(&g, 2);
    assert_eq!(parts.len(), 2);

    // Check that the partition separates the two cliques well
    let modularity = p.modularity(&g, &parts);
    assert!(modularity > 0.0, "modularity = {modularity}");
}

// ============ Theorem 10: Effective resistance is a metric ============
#[test]
fn test_theorem10_resistance_nonneg() {
    let g = Graph::path(4);
    let sp = GraphSparsifier::new();
    for i in 0..4 {
        for j in 0..4 {
            let r = sp.effective_resistance(&g, i, j);
            assert!(r >= -0.01, "R_{i},{j} = {r}");
        }
    }
}

#[test]
fn test_theorem10_resistance_identity() {
    let g = Graph::complete(4);
    let sp = GraphSparsifier::new();
    for i in 0..4 {
        let r = sp.effective_resistance(&g, i, i);
        assert!(r.abs() < 0.01, "R_{i},{i} = {r}");
    }
}

#[test]
fn test_theorem10_resistance_symmetry() {
    let g = Graph::path(4);
    let sp = GraphSparsifier::new();
    for i in 0..4 {
        for j in 0..4 {
            let rij = sp.effective_resistance(&g, i, j);
            let rji = sp.effective_resistance(&g, j, i);
            assert!((rij - rji).abs() < 0.01, "R_{i},{j} != R_{j},{i}");
        }
    }
}

#[test]
fn test_theorem10_resistance_triangle_inequality() {
    let g = Graph::path(5);
    let sp = GraphSparsifier::new();
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                let rij = sp.effective_resistance(&g, i, j);
                let rik = sp.effective_resistance(&g, i, k);
                let rkj = sp.effective_resistance(&g, k, j);
                assert!(
                    rij <= rik + rkj + 0.05,
                    "R_{i},{j} ({rij}) > R_{i},{k} ({rik}) + R_{k},{j} ({rkj})"
                );
            }
        }
    }
}

// ============ Theorem 11: Algebraic connectivity monotone under edge addition ============
#[test]
fn test_theorem11_monotone_edge_addition() {
    let mut g = Graph::path(5);
    let s1 = Spectrum::from_graph_laplacian(&g);
    let ac1 = s1.spectral_gap();

    g.add_edge(0, 4, 1.0); // add edge to make cycle
    let s2 = Spectrum::from_graph_laplacian(&g);
    let ac2 = s2.spectral_gap();

    assert!(ac2 >= ac1 - 0.01, "ac after ({ac2}) should be >= ac before ({ac1})");
}

#[test]
fn test_theorem11_monotone_adding_to_star() {
    let mut g = Graph::star(5);
    let s1 = Spectrum::from_graph_laplacian(&g);
    let ac1 = s1.spectral_gap();

    g.add_edge(1, 2, 1.0); // connect two leaves
    let s2 = Spectrum::from_graph_laplacian(&g);
    let ac2 = s2.spectral_gap();

    assert!(ac2 >= ac1 - 0.01, "ac after ({ac2}) should be >= ac before ({ac1})");
}

// ============ Theorem 12: Fiedler vector sign pattern gives optimal 2-way cut ============
#[test]
fn test_theorem12_fiedler_sign_cut() {
    let mut g = Graph::new(6);
    // Two clusters connected by single edge
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(0, 2, 1.0);
    g.add_edge(3, 4, 1.0);
    g.add_edge(4, 5, 1.0);
    g.add_edge(3, 5, 1.0);
    g.add_edge(2, 3, 1.0); // bridge

    let s = Spectrum::from_graph_laplacian(&g);
    let fv = s.fiedler_vector();

    // Nodes in same cluster should have same sign
    let sign = |v: f64| if v >= 0.0 { 1 } else { -1 };
    // {0,1,2} should be one sign, {3,4,5} should be opposite
    assert_eq!(sign(fv[0]), sign(fv[1]));
    assert_eq!(sign(fv[1]), sign(fv[2]));
    assert_eq!(sign(fv[3]), sign(fv[4]));
    assert_eq!(sign(fv[4]), sign(fv[5]));
    assert_ne!(sign(fv[0]), sign(fv[3]));
}

#[test]
fn test_theorem12_fiedler_path_cut() {
    let g = Graph::path(4);
    let s = Spectrum::from_graph_laplacian(&g);
    let fv = s.fiedler_vector();
    // Path 0-1-2-3: Fiedler vector should split at midpoint
    let sign = |v: f64| if v >= 0.0 { 1 } else { -1 };
    // One side: {0,1}, other side: {2,3}
    assert_eq!(sign(fv[0]), sign(fv[1]));
    assert_eq!(sign(fv[2]), sign(fv[3]));
    assert_ne!(sign(fv[0]), sign(fv[2]));
}

// ============ Additional theorem / property tests ============

#[test]
fn test_normalized_laplacian_eigenvalues_bounded() {
    // Eigenvalues of normalized Laplacian are in [0, 2]
    let g = Graph::path(6);
    let l_norm = g.normalized_laplacian();
    let (eigenvalues, _) = l_norm.eigendecomposition();
    for &e in &eigenvalues {
        assert!(e >= -0.01 && e <= 2.01, "normalized eigenvalue {e} out of [0,2]");
    }
}

#[test]
fn test_laplacian_row_sums_zero() {
    let g = Graph::complete(4);
    let l = g.laplacian();
    for i in 0..4 {
        let row_sum: f64 = l.data[i].iter().sum();
        assert!(row_sum.abs() < 1e-10, "row {i} sum = {row_sum}");
    }
}

#[test]
fn test_laplacian_is_psd() {
    let g = Graph::path(5);
    let l = g.laplacian();
    assert!(l.is_positive_semidefinite());
}

#[test]
fn test_complete_graph_uniform_pagerank() {
    // In a complete graph, all nodes should have equal PageRank
    let g = Graph::complete(5);
    let pr = PageRank::new();
    let ranks = pr.compute(&g, 0.85, 200);
    let expected = 1.0 / 5.0;
    for (i, &r) in ranks.iter().enumerate() {
        assert!(
            (r - expected).abs() < 0.01,
            "rank[{i}] = {r}, expected {expected}"
        );
    }
}

#[test]
fn test_agent_network_bottleneck_weak_link() {
    let mut g = Graph::new(4);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 0.1); // weak link
    g.add_edge(2, 3, 1.0);
    let agents = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    let net = AgentNetwork::new(agents, g);
    let (i, j, _) = net.bottleneck();
    // The bottleneck should be at the weak link (1,2)
    assert!((i == 1 && j == 2) || (i == 2 && j == 1), "bottleneck at ({i},{j})");
}

#[test]
fn test_conductance_decreases_with_larger_set() {
    let g = Graph::complete(6);
    let c = CheegerInvariant::new();
    let cond1 = c.conductance(&g, &[0]);
    let cond3 = c.conductance(&g, &[0, 1, 2]);
    // For K_6: cond({0}) = 5/5 = 1.0, cond({0,1,2}) = 9/15 = 0.6
    assert!(cond3 < cond1, "cond3={cond3} should be < cond1={cond1}");
}

#[test]
fn test_effective_resistance_path() {
    // In a path graph, effective resistance = graph distance (for unit weights)
    let g = Graph::path(4);
    let sp = GraphSparsifier::new();
    let r01 = sp.effective_resistance(&g, 0, 1);
    assert!((r01 - 1.0).abs() < 0.1, "R_01 = {r01}, expected 1.0");
    let r03 = sp.effective_resistance(&g, 0, 3);
    assert!((r03 - 3.0).abs() < 0.1, "R_03 = {r03}, expected 3.0");
}

#[test]
fn test_heat_kernel_positive_definite() {
    let g = Graph::path(4);
    let wave = GraphWave::new();
    let h = wave.heat_kernel(&g, 1.0);
    // Heat kernel should be positive definite
    assert!(h.is_positive_semidefinite());
}

#[test]
fn test_spectrum_serialization() {
    let g = Graph::complete(4);
    let s = Spectrum::from_graph_laplacian(&g);
    let json = serde_json::to_string(&s).unwrap();
    let s2: Spectrum = serde_json::from_str(&json).unwrap();
    assert_eq!(s2.eigenvalues.len(), 4);
    for i in 0..4 {
        assert!((s.eigenvalues[i] - s2.eigenvalues[i]).abs() < 1e-10);
    }
}
