# lau-spectral-graph-agent

Spectral graph theory library for analyzing agent network structure via eigenvalues of graph operators.

## Components

- **`Graph`** — adjacency-list weighted graph with Laplacian variants (standard, normalized, random walk)
- **`DenseMatrix`** — dense matrix with eigendecomposition (Jacobi iteration), inverse, matrix exponential
- **`Spectrum`** — eigenvalue decomposition with spectral gap, Fiedler vector, entropy, energy
- **`SpectralPartitioner`** — k-way spectral clustering with eigengap heuristic and Newman-Girvan modularity
- **`CheegerInvariant`** — discrete Cheeger constant with inequality bounds and conductance
- **`GraphWave`** — spectral graph wavelets: heat kernel, multi-scale embeddings, characteristic functions
- **`AgentNetwork`** — agent communication graph analysis: algebraic connectivity, bottlenecks, robustness, broadcast ordering
- **`PageRank`** — power-iteration PageRank with personalized and trust variants
- **`GraphSparsifier`** — effective resistance, spanners, spectral sparsification

## Theorems Verified (91 tests)

1. Multiplicity of eigenvalue 0 = number of connected components
2. Cheeger inequality: λ₁/2 ≤ h(G) ≤ √(2λ₁)
3. Complete graph K_n: eigenvalues n (mult n-1) and 0
4. Path graph P_n: eigenvalues 2(1−cos(πk/n))
5. Cycle graph C_n: eigenvalues 2(1−cos(2πk/n))
6. Star graph S_n: λ₁ = 1
7. PageRank sums to 1
8. Heat kernel trace: tr(Hₜ) = Σ e^{−tλᵢ}
9. Spectral partitioning finds good cuts
10. Effective resistance is a metric
11. Algebraic connectivity is monotone under edge addition
12. Fiedler vector gives optimal 2-way cut

## Usage

```rust
use lau_spectral_graph_agent::*;

let g = Graph::complete(5);
let spectrum = Spectrum::from_graph_laplacian(&g);
println!("Algebraic connectivity: {}", spectrum.spectral_gap());
println!("Connected: {}", spectrum.is_connected());
println!("Components: {}", spectrum.number_of_components());
```

## License

MIT
