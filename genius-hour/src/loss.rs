use nalgebra::DMatrix;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy, // Assumes predictions are probabilities (e.g., from Softmax)
}

impl LossFunction {
    pub fn calculate(&self, predictions: &DMatrix<f32>, targets: &DMatrix<f32>) -> f32 {
        assert_eq!(predictions.shape(), targets.shape(), "Predictions and targets shape mismatch for loss calculation.");
        let batch_size = predictions.nrows() as f32;
        match self {
            LossFunction::MeanSquaredError => {
                (predictions - targets).map(|x| x * x).sum() / (2.0 * batch_size) // 0.5 * MSE
            }
            LossFunction::CrossEntropy => {
                // Add epsilon to prevent log(0)
                let epsilon = f32::EPSILON;
                let clipped_predictions = predictions.map(|p| p.max(epsilon).min(1.0 - epsilon));
                - (targets.component_mul(&clipped_predictions.map(|p| p.ln()))).sum() / batch_size
            }
        }
    }

    // Derivative of the loss function w.r.t. the predictions (network's output activations)
    pub fn derivative(&self, predictions: &DMatrix<f32>, targets: &DMatrix<f32>) -> DMatrix<f32> {
        assert_eq!(predictions.shape(), targets.shape(), "Predictions and targets shape mismatch for loss derivative.");
        let batch_size = predictions.nrows() as f32;
        match self {
            LossFunction::MeanSquaredError => {
                (predictions - targets) / batch_size
            }
            LossFunction::CrossEntropy => {
                // Add epsilon to prevent division by zero
                let epsilon = f32::EPSILON;
                let clipped_predictions = predictions.map(|p| p.max(epsilon).min(1.0 - epsilon));
                // dL/dp = - (targets / predictions)
                // This derivative is w.r.t. p (network output).
                // If the last layer is Softmax, the combined derivative dL/dz = p - y is simpler.
                // This function returns dL/dp. The network's backprop logic handles combining it.
                 -targets.component_div(&clipped_predictions) / batch_size
            }
        }
    }
}