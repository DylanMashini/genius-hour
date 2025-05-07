use nalgebra::DMatrix;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    Linear,
    Sigmoid,
    ReLU,
    Softmax,
}

impl ActivationFunction {
    pub fn activate(&self, z: &DMatrix<f32>) -> DMatrix<f32> {
        match self {
            ActivationFunction::Linear => z.clone(),
            ActivationFunction::Sigmoid => z.map(|val| 1.0 / (1.0 + (-val).exp())),
            ActivationFunction::ReLU => z.map(|val| val.max(0.0)),
            ActivationFunction::Softmax => {
                let max_val = z.max();
                let exp_z = z.map(|val| (val - max_val).exp());
                let sum_exp_z = exp_z.sum();
                if z.ncols() == 1 || z.nrows() == 1 {
                    exp_z / sum_exp_z
                } else {
                    let mut output = DMatrix::zeros(z.nrows(), z.ncols());
                    for r in 0..z.nrows() {
                        let row = z.row(r);
                        let row_max = row.max();
                        let exp_row = row.map(|val| (val - row_max).exp());
                        let sum_exp_row = exp_row.sum();
                        output.set_row(r, &(exp_row / sum_exp_row));
                    }
                    output
                }
            }
        }
    }

    // Derivative of activation function w.r.t. its input z
    // For Softmax, this is typically combined with CrossEntropy loss,
    // where dLoss/dZ = P - Y. So this derivative might not be directly used
    // if that optimization is applied. Here we provide the diagonal of the Jacobian.
    pub fn derivative(&self, z: &DMatrix<f32>) -> DMatrix<f32> {
        match self {
            ActivationFunction::Linear => DMatrix::from_element(z.nrows(), z.ncols(), 1.0),
            ActivationFunction::Sigmoid => {
                let s = self.activate(z);
                s.component_mul(&s.map(|val| 1.0 - val))
            }
            ActivationFunction::ReLU => z.map(|val| if val > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Softmax => {
                // This is simplified: derivative of softmax_i w.r.t z_i is p_i * (1 - p_i)
                // The full Jacobian is more complex: J_ij = p_i * (delta_ij - p_j)
                // For use in backprop element-wise multiplication, this (p_i * (1-p_i)) is common if loss
                // derivative is already dL/dp.
                // However, the common optimization d(CrossEntropy)/dZ = P - Y bypasses this.
                // We'll assume this derivative function is called when that optimization isn't used.
                let p = self.activate(z);
                p.component_mul(&p.map(|val| 1.0 - val))
            }
        }
    }
}