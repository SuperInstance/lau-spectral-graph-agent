# lau-spectral-graph-agent

> The eigenvalues of a graph's Laplacian encode everything about its structure. The Fiedler vector is the graph's spine — spectral analysis reads the skeleton beneath the noise.

[![tests](https://img.shields.io/badge/tests-91-green)]()
[![license](https://img.shields.io/badge/license-MIT-blue)]()

## What This Does

This crate implements **spectral graph theory** — the study of graphs through the eigenvalues and eigenvectors of their matrix representations — with direct application to **agent network analysis**.

It provides:

- **4 Laplacian variants** (combinatorial, normalized, random-walk, signless) for capturing different structural properties
- **Spectral partitioning** via k-means on eigenvector embeddings (Shi–Malik / Ng–Jordan–Weiss style)
- **Cheeger constant** computation with exact enumeration (small graphs) and Fiedler sweep (large graphs)
- **Heat kernel wavelets** for multi-scale graph analysis
- **PageRank** (standard, personalized, and TrustRank) via power iteration
- **Spectral sparsification** using effective resistance sampling (Spielman–Srivastava)
- **Agent network** diagnostics: algebraic connectivity, bottleneck detection, robustness, broadcast ordering

Every structure is `serde`-serializable with zero unsafe code.

## The Key Idea

A graph's **Laplacian matrix** `L = D - A` has eigenvalues `0 = λ₀ ≤ λ₁ ≤ ⋯ ≤ λₙ₋₁` that reveal deep structural information:

| Eigenvalue | What it tells you |
|-----------|-------------------|
| `λ₁ = 0` (multiplicity) | Number of connected components |
| `λ₁ > 0` | Graph is connected (algebraic connectivity) |
| `λ₁` (magnitude) | How well-connected (larger = harder to disconnect) |
| Fiedler vector (eigvec of `λ₁`) | The spectral cut — how to partition the graph |
| `λₙ₋₁` | Maximum "bottlenecking" behavior |
| `λₙ₋₁ / λ₁` | Spectral condition number — governs mixing time |

The **Cheeger inequality** `λ₁/2 ≤ h(G) ≤ √(2λ₁)` connects the spectral gap to combinatorial cuts, and this crate verifies it on every graph.

## Install

```bash
cargo add lau-spectral-graph-agent
```

## Quick Start

```rust
use lau_spectral_graph_agent::*;

// Build a graph: two triangles connected by a weak bridge
let mut g = Graph::new(6);
g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0); // clique 1
g.add_edge(3, 4, 1.0); g.add_edge(4, 5, 1.0); g.add_edge(3, 5, 1.0); // clique 2
g.add_edge(2, 3, 0.1); // weak bridge

// Spectral analysis
let spectrum = Spectrum::from_graph_laplacian(&g);
println!("Connected: {}", spectrum.is_connected());           // true
println!("Algebraic connectivity: {:.4}", spectrum.spectral_gap());
println!("Fiedler vector: {:?}", spectrum.fiedler_vector());

// Partition into 2 communities
let partitioner = SpectralPartitioner::new();
let parts = partitioner.partition(&g, 2);
println!("Communities: {:?}", parts);
println!("Modularity: {:.4}", partitioner.modularity(&g, &parts));

// Cheeger constant and inequality
let cheeger = CheegerInvariant::new();
let h = cheeger.cheeger_constant(&g);
let (lower, upper) = cheeger.cheeger_from_eigenvalue(spectrum.spectral_gap());
println!("h(G) = {:.4}, bounds: [{:.4}, {:.4}]", h, lower, upper);

// Agent network analysis
let net = AgentNetwork::new(
    vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into(), "f".into()],
    g,
);
println!("Bottleneck: {:?}", net.bottleneck());
println!("Broadcast order: {:?}", net.optimal_broadcast_order());
```

## API Reference

### Graph

Weighted undirected graph with named constructors and Laplacian variants.

| Method | Description |
|--------|-------------|
| `new(n)` | Empty graph with `n` vertices |
| `add_edge(i, j, w)` | Add weighted undirected edge |
| `adjacency_matrix()` | Build the adjacency matrix `A` |
| `degree_matrix()` | Build the diagonal degree matrix `D` |
| `laplacian()` | Combinatorial Laplacian `L = D - A` |
| `normalized_laplacian()` | Symmetric normalized `L_sym = D^{-1/2} L D^{-1/2}` |
| `random_walk_laplacian()` | Random-walk normalized `L_rw = D^{-1} L` |
| `degree(i)` / `total_volume()` | Degree queries |
| `has_edge(i, j)` / `edge_weight(i, j)` | Edge queries |
| `complete(n)` / `path(n)` / `cycle(n)` / `star(n)` | Named graph constructors |
| `induced_subgraph(vertices)` | Subgraph restricted to vertex set |

### Spectrum

Eigenvalue decomposition of graph matrices.

| Method | Description |
|--------|-------------|
| `from_graph_laplacian(graph)` | Eigendecomposition of `L` |
| `from_graph_normalized_laplacian(graph)` | Eigendecomposition of `L_sym` |
| `from_matrix(matrix)` | Eigendecomposition of any dense matrix |
| `spectral_gap()` | Second smallest eigenvalue `λ₁` (algebraic connectivity) |
| `is_connected()` | `true` iff `λ₁ > 0` |
| `number_of_components()` | Multiplicity of eigenvalue 0 |
| `fiedler_vector()` | Eigenvector of `λ₁` (the spectral cut) |
| `spectral_radius()` | Largest `|λ|` |
| `energy()` | Sum of `|λ|` (graph energy) |
| `entropy()` | Spectral entropy `-Σ pᵢ ln pᵢ` |

### SpectralPartitioner

K-way spectral clustering via normalized Laplacian + k-means.

| Method | Description |
|--------|-------------|
| `partition(graph, k)` | Split graph into `k` communities |
| `optimal_k(graph, max_k)` | Eigengap heuristic for optimal `k` |
| `modularity(graph, partition)` | Newman–Girvan modularity `Q` |

### CheegerInvariant

The Cheeger constant `h(G) = min_S |∂S|/min(|S|, |V\S|)`.

| Method | Description |
|--------|-------------|
| `cheeger_constant(graph)` | Exact (n ≤ 20) or Fiedler sweep (n > 20) |
| `cheeger_from_eigenvalue(λ)` | Bounds: `(λ/2, √(2λ))` |
| `conductance(graph, set)` | `|∂S|/vol(S)` |

### GraphWave

Heat kernel wavelets for multi-scale graph analysis.

| Method | Description |
|--------|-------------|
| `heat_kernel(graph, t)` | `H_t = e^{-tL}` (diffusion matrix) |
| `wavelet(graph, node, scale)` | Spectral wavelet centered at a node |
| `multi_scale_embedding(graph, scales)` | Per-node features at multiple scales |
| `characteristic_function(graph, t)` | Heat kernel trace per node |

### AgentNetwork

Agent network with spectral diagnostics.

| Method | Description |
|--------|-------------|
| `new(agents, graph)` | Create from agent names and topology |
| `algebraic_connectivity()` | `λ₁` of the network |
| `bottleneck()` | Weakest link (by Fiedler cut) |
| `robustness()` | Connectivity after removing central agent |
| `bottleneck_agents()` | Agents on the spectral cut boundary |
| `optimal_broadcast_order()` | Eigenvector centrality ordering |

### PageRank

PageRank via power iteration with teleportation.

| Method | Description |
|--------|-------------|
| `compute(graph, damping, iterations)` | Standard PageRank |
| `personalized(graph, probs, damping)` | Personalized PageRank |
| `trust_rank(graph, trusted, damping)` | TrustRank from seed nodes |

### GraphSparsifier

Spectral sparsification via effective resistance sampling.

| Method | Description |
|--------|-------------|
| `effective_resistance(graph, i, j)` | `R_ij = (eᵢ - eⱼ)ᵀ L⁺ (eᵢ - eⱼ)` |
| `spanner(graph, stretch)` | Greedy spanner construction |
| `spectral_sparsify(graph, ε)` | Spielman–Srivastava style sparsification |

### DenseMatrix

General-purpose dense matrix with linear algebra operations.

| Method | Description |
|--------|-------------|
| `zeros(r, c)` / `identity(n)` | Constructors |
| `eigendecomposition()` | Full eigenvalue/eigenvector decomposition |
| `is_symmetric()` / `is_positive_semidefinite()` | Matrix properties |
| `add` / `sub` / `mul` / `scale` | Arithmetic |
| `trace()` / `transpose()` / `exp_symmetric()` | Operations |

## How It Works

### Eigendecomposition Pipeline

```
Graph → Laplacian (4 variants) → DenseMatrix → QR iteration → eigenvalues + eigenvectors
         ↓
    Spectrum → spectral gap, Fiedler vector, entropy, energy
         ↓
    SpectralPartitioner → k-means on eigenvector embeddings
    CheegerInvariant → Fiedler sweep for cut quality
    GraphWave → heat kernel H_t = e^{-tL}
```

### Partitioning Algorithm (Shi–Malik / Ng–Jordan–Weiss)

1. Compute the normalized Laplacian `L_sym = D^{-1/2} L D^{-1/2}`
2. Extract the `k` smallest eigenvectors (skip the constant first one)
3. Embed each node into `ℝᵏ` using eigenvector coordinates
4. Run k-means clustering on the embeddings
5. Evaluate quality via Newman–Girvan modularity

### Spectral Sparsification (Spielman–Srivastava)

1. Compute effective resistance `R_ij` for each edge using the Laplacian pseudoinverse
2. Sample edges with probability proportional to `w_ij · R_ij`
3. Rescale sampled edges to preserve spectral properties
4. Add back edges to ensure connectivity

## The Math

### Core Theorems

| Theorem | Statement | Verified in tests |
|---------|-----------|-------------------|
| **Algebraic connectivity** | `λ₁ > 0 ⟺ G is connected` | ✅ Complete vs disconnected graphs |
| **Fiedler cut** | The Fiedler vector's sign pattern gives an approximate min-cut | ✅ Two-clique separation |
| **Cheeger inequality** | `λ₁/2 ≤ h(G) ≤ √(2λ₁)` | ✅ Verified on path, complete, barbell |
| **PageRank convergence** | Power iteration converges; ranks sum to 1 | ✅ Sum-to-one, center > leaves in star |
| **Effective resistance** | `R_ij` satisfies triangle inequality, symmetry, non-negativity | ✅ All three properties |
| **Spanner connectivity** | Greedy spanner preserves connectivity | ✅ Connected after sparsification |
| **Spectral sparsification** | Sparse graph maintains connectivity | ✅ Connected after sampling |
| **Heat kernel** | `tr(H_t) = Σ e^{-tλᵢ}` | ✅ Trace matches eigenvalue sum |
| **Euler's formula** | Complete graph K₃: eigenvalues are `0, 3, 3` | ✅ Energy = 6 |
| **Modularity bounds** | Single community has Q ≈ 0; separated components have Q > 0 | ✅ Both cases |

### Laplacian Variants

| Variant | Formula | Properties |
|---------|---------|------------|
| Combinatorial | `L = D - A` | PSD, row sums = 0 |
| Normalized | `L_sym = D^{-1/2} L D^{-1/2}` | Symmetric, eigenvalues in [0, 2] |
| Random walk | `L_rw = D^{-1} L` | Left-stochastic transition matrix |

### Effective Resistance

For a graph with Laplacian `L`, the effective resistance between nodes `i` and `j` is:

```
R_ij = (eᵢ - eⱼ)ᵀ L⁺ (eᵢ - eⱼ)
```

where `L⁺` is the Moore–Penrose pseudoinverse. This is a **distance metric** on graphs (non-negative, symmetric, triangle inequality). For the complete graph `Kₙ`: `R_ij = 2/n`.

## Testing

**91 tests** across 9 modules covering:

- **Graph**: Laplacian construction, named graphs, serde, degree
- **Matrix**: Eigendecomposition, symmetry, PSD, arithmetic
- **Spectrum**: Connected/disconnected, spectral gap, Fiedler vector, entropy, energy
- **Partitioner**: Two-cluster separation, optimal k heuristic, modularity
- **Cheeger**: Exact constant computation, inequality verification, conductance
- **Wavelets**: Heat kernel trace, symmetry, localization, multi-scale embedding
- **Agent Network**: Connectivity, bottleneck, robustness, broadcast ordering, serde
- **PageRank**: Sum to one, star center dominance, path middle nodes, personalization, TrustRank
- **Sparsifier**: Effective resistance properties, spanner connectivity, spectral sparsification

## License

MIT
