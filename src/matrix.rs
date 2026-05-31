use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl DenseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i][i] = 1.0;
        }
        m
    }

    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i][j] = v;
    }

    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.data[i][j] - self.data[j][i]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_positive_semidefinite(&self) -> bool {
        if !self.is_symmetric() {
            return false;
        }
        // Check via eigenvalues — but we use Cholesky-like check for efficiency
        // For small matrices, compute eigenvalues
        let eigenvals = self.eigenvalues();
        eigenvals.iter().all(|&v| v > -1e-10)
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut result = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> Self {
        let mut result = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * s;
            }
        }
        result
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i][j] * v[j];
            }
        }
        result
    }

    /// Compute eigenvalues using QR iteration with shifts
    pub fn eigenvalues(&self) -> Vec<f64> {
        assert_eq!(self.rows, self.cols, "Matrix must be square for eigenvalues");
        let n = self.rows;
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![self.data[0][0]];
        }

        // Symmetrize if nearly symmetric (use symmetric algorithm for real eigenvalues)
        let symmetric = self.is_symmetric();
        if symmetric {
            self.symmetric_eigenvalues()
        } else {
            // For non-symmetric, still use QR iteration (may miss complex eigenvalues)
            self.qr_eigenvalues()
        }
    }

    fn symmetric_eigenvalues(&self) -> Vec<f64> {
        let n = self.rows;
        // Tridiagonalize via Householder, then QR iteration
        let (diag, off_diag) = self.householder_tridiag();
        Self::tridiag_qr_eigenvalues(diag, off_diag, n)
    }

    fn householder_tridiag(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.rows;
        let mut a = self.data.clone();
        let mut diag = vec![0.0; n];
        let mut off_diag = vec![0.0; n];

        let mut v = vec![vec![0.0; n]; n];

        for k in 0..n.saturating_sub(2) {
            // Extract column below diagonal
            let mut x = vec![0.0; n - k - 1];
            for i in 0..n - k - 1 {
                x[i] = a[k + 1 + i][k];
            }

            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_x < 1e-15 {
                continue;
            }

            let alpha = if x[0] >= 0.0 { -norm_x } else { norm_x };
            let mut u = x.clone();
            u[0] -= alpha;
            let norm_u = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_u < 1e-15 {
                continue;
            }
            for item in &mut u {
                *item /= norm_u;
            }

            // P = I - 2*u*u^T
            // A = P * A * P
            // Apply from left: pA = A - 2*u*(u^T*A)
            let size = n - k - 1;
            // Compute u^T * A (submatrix)
            let mut ut_a = vec![0.0; n];
            for j in 0..n {
                let mut s = 0.0;
                for i in 0..size {
                    s += u[i] * a[k + 1 + i][j];
                }
                ut_a[j] = s;
            }

            for i in 0..size {
                for j in 0..n {
                    a[k + 1 + i][j] -= 2.0 * u[i] * ut_a[j];
                }
            }

            // Apply from right: A * p = A - 2*(A*u)*u^T
            let mut a_u = vec![0.0; n];
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..size {
                    s += a[i][k + 1 + j] * u[j];
                }
                a_u[i] = s;
            }

            for i in 0..n {
                for j in 0..size {
                    a[i][k + 1 + j] -= 2.0 * a_u[i] * u[j];
                }
            }

            // Store for eigenvector accumulation (simplified — we skip eigenvectors here)
            for i in 0..size {
                v[k][k + 1 + i] = u[i];
            }
        }

        // Extract tridiagonal
        for i in 0..n {
            diag[i] = a[i][i];
            if i + 1 < n {
                off_diag[i] = a[i][i + 1];
            }
        }

        (diag, off_diag)
    }

    fn tridiag_qr_eigenvalues(mut diag: Vec<f64>, mut off_diag: Vec<f64>, n: usize) -> Vec<f64> {
        // Implicit QR with Wilkinson shift
        let max_iter = 100 * n;
        let mut m = n;
        let tol = 1e-12;

        for _ in 0..max_iter {
            if m <= 1 {
                break;
            }

            // Check for convergence of off-diagonal elements
            while m > 1 && off_diag[m - 2].abs() <= tol * (diag[m - 1].abs() + diag[m - 2].abs()) {
                m -= 1;
            }
            if m <= 1 {
                break;
            }

            // Wilkinson shift
            let d = (diag[m - 1] - diag[m - 2]) / 2.0;
            let shift = diag[m - 1] - off_diag[m - 2].powi(2) / (d + d.signum() * (d * d + off_diag[m - 2].powi(2)).sqrt());

            // Implicit QR step (chase the bulge)
            let mut x = diag[0] - shift;
            let mut z = off_diag[0];

            for k in 0..m - 1 {
                let r = (x * x + z * z).sqrt();
                if r < 1e-30 {
                    break;
                }
                let c = x / r;
                let s = -z / r;

                // Apply Givens rotation
                let w = if k > 0 { off_diag[k - 1] } else { 0.0 };
                let d1 = diag[k];
                let e = off_diag[k];
                let d2 = diag[k + 1];

                if k > 0 {
                    off_diag[k - 1] = c * w - s * z;
                }

                diag[k] = c * c * d1 - 2.0 * c * s * e + s * s * d2;
                diag[k + 1] = s * s * d1 + 2.0 * c * s * e + c * c * d2;
                off_diag[k] = c * s * (d1 - d2) + (c * c - s * s) * e;

                if k + 2 < m {
                    z = -s * off_diag[k + 1];
                    off_diag[k + 1] = c * off_diag[k + 1];
                    x = off_diag[k];
                }
            }
        }

        diag.sort_by(|a, b| a.partial_cmp(b).unwrap());
        diag
    }

    fn qr_eigenvalues(&self) -> Vec<f64> {
        let n = self.rows;
        let mut a = self.data.clone();

        for _ in 0..200 * n {
            // QR decomposition via Householder
            let (q, r) = Self::qr_decompose(&a, n);
            // A = R * Q
            for i in 0..n {
                for j in 0..n {
                    a[i][j] = 0.0;
                    for k in 0..n {
                        a[i][j] += r.data[i][k] * q.data[k][j];
                    }
                }
            }

            // Check convergence
            let mut off_diag_sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off_diag_sum += a[i][j] * a[i][j];
                    }
                }
            }
            if off_diag_sum < 1e-20 {
                break;
            }
        }

        let mut eigenvals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        eigenvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eigenvals
    }

    fn qr_decompose(a: &[Vec<f64>], n: usize) -> (Self, Self) {
        let mut r = Self::from_vec(a.to_vec());
        let mut q = Self::identity(n);

        for k in 0..n.saturating_sub(1) {
            // Householder for column k
            let mut x = vec![0.0; n - k];
            for i in k..n {
                x[i - k] = r.data[i][k];
            }

            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_x < 1e-15 {
                continue;
            }

            let alpha = if x[0] >= 0.0 { -norm_x } else { norm_x };
            x[0] -= alpha;
            let norm_u = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_u < 1e-15 {
                continue;
            }
            for item in &mut x {
                *item /= norm_u;
            }

            // R = H * R
            for j in k..n {
                let dot: f64 = x.iter().zip(k..n).map(|(u, i)| u * r.data[i][j]).sum();
                for i in k..n {
                    r.data[i][j] -= 2.0 * x[i - k] * dot;
                }
            }

            // Q = Q * H
            for j in 0..n {
                let dot: f64 = x.iter().zip(k..n).map(|(u, i)| u * q.data[j][i]).sum();
                for i in k..n {
                    q.data[j][i] -= 2.0 * x[i - k] * dot;
                }
            }
        }

        (q, r)
    }

    /// Full eigendecomposition for symmetric matrices
    pub fn eigendecomposition(&self) -> (Vec<f64>, Vec<Vec<f64>>) {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        if n == 0 {
            return (vec![], vec![]);
        }
        if n == 1 {
            return (vec![self.data[0][0]], vec![vec![1.0]]);
        }

        // Use Jacobi iteration for symmetric matrices (more numerically stable for eigenvectors)
        let mut a = self.data.clone();
        let mut v = Self::identity(n).data;

        let max_iter = 100 * n * n;
        for _ in 0..max_iter {
            // Find largest off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < 1e-12 {
                break;
            }

            // Compute rotation angle
            let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
            };
            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation
            let app = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
            let aqq = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
            a[p][p] = app;
            a[q][q] = aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for r in 0..n {
                if r != p && r != q {
                    let arp = c * a[r][p] + s * a[r][q];
                    let arq = -s * a[r][p] + c * a[r][q];
                    a[r][p] = arp;
                    a[p][r] = arp;
                    a[r][q] = arq;
                    a[q][r] = arq;
                }
            }

            // Update eigenvectors
            for r in 0..n {
                let vrp = c * v[r][p] + s * v[r][q];
                let vrq = -s * v[r][p] + c * v[r][q];
                v[r][p] = vrp;
                v[r][q] = vrq;
            }
        }

        let mut eigenvals: Vec<(f64, usize)> = (0..n).map(|i| (a[i][i], i)).collect();
        eigenvals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let eigenvalues: Vec<f64> = eigenvals.iter().map(|(v, _)| *v).collect();
        let eigenvectors: Vec<Vec<f64>> = eigenvals
            .iter()
            .map(|(_, idx)| (0..n).map(|r| v[r][*idx]).collect())
            .collect();

        (eigenvalues, eigenvectors)
    }

    /// Compute inverse (for small matrices)
    pub fn inverse(&self) -> Option<Self> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        // Gauss-Jordan elimination
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = self.data[i][j];
            }
            aug[i][n + i] = 1.0;
        }

        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                return None; // Singular
            }

            // Swap rows
            aug.swap(col, max_row);

            // Scale pivot row
            let pivot = aug[col][col];
            for j in 0..2 * n {
                aug[col][j] /= pivot;
            }

            // Eliminate column
            for row in 0..n {
                if row != col {
                    let factor = aug[row][col];
                    for j in 0..2 * n {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }

        let mut inv = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.data[i][j] = aug[i][n + j];
            }
        }
        Some(inv)
    }

    /// Solve Ax = b
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        let inv = self.inverse()?;
        Some(inv.mul_vec(b))
    }

    /// Matrix exponentiation via eigendecomposition (symmetric only)
    pub fn exp_symmetric(&self) -> Self {
        let (eigenvalues, eigenvectors) = self.eigendecomposition();
        let n = self.rows;

        // A = V * diag(λ) * V^T
        // exp(A) = V * diag(exp(λ)) * V^T
        let mut result = Self::zeros(n, n);
        for k in 0..n {
            let exp_lambda = eigenvalues[k].exp();
            for i in 0..n {
                for j in 0..n {
                    result.data[i][j] += exp_lambda * eigenvectors[k][i] * eigenvectors[k][j];
                }
            }
        }
        result
    }

    pub fn trace(&self) -> f64 {
        (0..self.rows.min(self.cols)).map(|i| self.data[i][i]).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let m = DenseMatrix::identity(3);
        assert!(m.is_symmetric());
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 0.0);
    }

    #[test]
    fn test_mul() {
        let a = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = DenseMatrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = a.mul(&b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_inverse() {
        let a = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let inv = a.inverse().unwrap();
        let product = a.mul(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_eigenvalues_identity() {
        let m = DenseMatrix::identity(3);
        let eigs = m.eigenvalues();
        for e in &eigs {
            assert!((e - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        let m = DenseMatrix::from_vec(vec![vec![3.0, 0.0], vec![0.0, 7.0]]);
        let eigs = m.eigenvalues();
        assert!((eigs[0] - 3.0).abs() < 1e-8);
        assert!((eigs[1] - 7.0).abs() < 1e-8);
    }

    #[test]
    fn test_is_positive_semidefinite() {
        let m = DenseMatrix::identity(3);
        assert!(m.is_positive_semidefinite());
        let neg = DenseMatrix::from_vec(vec![vec![-1.0, 0.0], vec![0.0, -1.0]]);
        assert!(!neg.is_positive_semidefinite());
    }

    #[test]
    fn test_serde() {
        let m = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let json = serde_json::to_string(&m).unwrap();
        let m2: DenseMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.get(0, 1), 2.0);
    }
}
