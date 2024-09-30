use nalgebra::DMatrix;

const LEARNING_RATE: f64 = 1.0;
const BIAS: f64 = 1.0;

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn heaviside(x: f64) -> f64 {
    if x > 0.0 {
        return 1.0
    } else {
        return 0.0
    }
}


pub struct Perceptron {
    weights: DMatrix<f64>,
}

impl Perceptron {
    pub fn new(input_dim: usize) -> Self {
        Self {
            // Right now it is a Column Matrix with each layer's weight being in one row, and the bias being in the final row.
            weights: DMatrix::new_random(input_dim + 1, 1),
        }
    }

    fn step(&mut self, X: DMatrix<f64>, y: DMatrix<f64>) {
        let ones = DMatrix::from_element(X.nrows(), 1, BIAS);
        let X_bias = ones.hstack();

    }

    fn fit(&mut self, X: DMatrix<f64>, y: DMatrix<f64>) {

    }
}

#[cfg(test)]
mod test {
    use nalgebra::DMatrix;
    use crate::models::perceptron::Perceptron;

    #[test]
    fn perceptron_or() {
        // Each row is a sample
        let X = DMatrix::from_row_slice(4, 2, &[
            1.0, 1.0,
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
        ]);

        let Y = DMatrix::from_row_slice(4, 1, &[
            1.0,
            1.0,
            1.0,
            0.0
        ]);

        let mut model = Perceptron::new(2);
        model.fit(X, Y);
        
    }
}
