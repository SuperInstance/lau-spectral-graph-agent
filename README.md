# lau-spectral-graph-agent

The eigenvalues of a graph's Laplacian encode everything about its structure. The Fiedler vector is the graph's spine — spectral analysis reads the skeleton beneath the noise.

This crate implements four Laplacians, PageRank, heat kernels, graph wavelets, and spectral sparsification.

## The math in 60 seconds

The **graph Laplacian** L = D - A (degree matrix minus adjacency) has eigenvalues 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ. The second-smallest eigenvalue λ₂ (the **Fiedler value**) controls connectivity, mixing time, and synchronizability.

Key tools:

- **4 Laplacians:** combinatorial (L = D-A), normalized (L = I - D^{-1/2}AD^{-1/2}), random walk (L = I - D⁻¹A), signless (Q = D+A)
- **Fiedler vector:** eigenvector for λ₂ — threshold it to get an approximate min-cut
- **PageRank:** π = α(I - (1-α)D⁻¹A)⁻¹v — the steady-state of a random surfer
- **Heat kernel:** Hₜ = e^{-tL} — diffusion on the graph, reveals multi-scale structure
- **Spectral sparsification:** keep O(n/ε²) edges preserving Laplacian quadratic form within (1±ε)
- **Cheeger inequality:** λ₂/2 ≤ h(G) ≤ √(2λ₂) — spectral gap bounds the best cut

References: Chung, *Spectral Graph Theory* (1997); Spielman, *Spectral Graph Theory and its Applications* (2012)

## Quick start

```rust
use lau_spectral_graph_agent::{Graph, Spectrum, PageRank, HeatKernel};

// Build a graph
let graph = Graph::erdos_renyi(100, 0.05);

// Compute spectrum of normalized Laplacian
let spectrum = Spectrum::normalized_laplacian(&graph);
let fiedler = spectrum.fiedler_value();   // λ₂
let gap = spectrum.spectral_gap();

// Fiedler cut — bipartition the graph
let cut = spectrum.fiedler_cut(&graph);

// PageRank with α = 0.85
let pr = PageRank::compute(&graph, 0.85, 100);
let top_nodes = pr.top_k(10);

// Heat kernel diffusion for t = 1.0
let hk = HeatKernel::compute(&graph, 1.0, &source_node);

// Spectral sparsification (keep 10% of edges)
let sparse = graph.sparsify(0.1);
```

## Key types

| Type | What it is |
|------|-----------|
| `Graph` | Weighted undirected graph with construction methods |
| `Spectrum` | Eigenvalues/eigenvectors of 4 Laplacian variants |
| `Cheeger` | Cheeger constant and approximate min-cut |
| `PageRank` | The steady-state random surfer distribution |
| `HeatKernel` | Diffusion operator Hₜ = e^{-tL} |
| `Wavelet` | Spectral graph wavelets for multi-scale analysis |

## Contributing

[Open an issue](https://github.com/SuperInstance/lau-spectral-graph-agent/issues) or PR. Good directions:

- Directed spectral theory
- Local spectral methods (personalized PageRank)
- Applications to graph neural networks
