use nalgebra::{DMatrix, DVector};
use rand_distr::{Normal, Distribution};
use crate::activation::ActivationFunction;

pub struct DenseLayer {
    pub weights: DMatrix<f32>,    // Shape: (input_size, output_size)
    pub biases: DVector<f32>,     // Shape: (output_size, 1) -> DVector is a column vector
    pub activation_fn: ActivationFunction,

    // Cache for backpropagation
    input_cache: DMatrix<f32>,    // Input to this layer (A from prev layer or X)
    pub z_cache: DMatrix<f32>,    // Weighted sum + bias (input to activation function), made public
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation_fn: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        
        let std_dev = match activation_fn {
            ActivationFunction::ReLU => (2.0 / input_size as f32).sqrt(),
            _ => (1.0 / input_size as f32).sqrt(), 
        };
        let normal = Normal::new(0.0, std_dev).unwrap();

        let weights_data = (0..input_size * output_size)
            .map(|_| normal.sample(&mut rng))
            .collect::<Vec<f32>>();
        let weights = DMatrix::from_vec(input_size, output_size, weights_data);
        
        let biases = DVector::zeros(output_size); // DVector is (output_size, 1)

        DenseLayer {
            weights,
            biases,
            activation_fn,
            input_cache: DMatrix::zeros(0, 0), 
            z_cache: DMatrix::zeros(0, 0),     
        }
    }

    pub fn forward(&mut self, input: &DMatrix<f32>) -> DMatrix<f32> {
        assert_eq!(input.ncols(), self.weights.nrows(), 
            "FORWARD: Input columns ({}) must match weight rows ({}). Input dims: {}x{}, Weight dims: {}x{}", 
            input.ncols(), self.weights.nrows(), 
            input.nrows(), input.ncols(), 
            self.weights.nrows(), self.weights.ncols());
        self.input_cache = input.clone();
        
        let z_linear = input * &self.weights; // (batch_size, output_size)
        
        let bias_row_vector = self.biases.transpose(); // (1, output_size), type RowDVector<f32>

        // Explicitly compute z_linear + bias_row_vector row by row
        let mut z_biased = DMatrix::zeros(z_linear.nrows(), z_linear.ncols());
        for r_idx in 0..z_linear.nrows() {
            let row_sum = z_linear.row(r_idx) + &bias_row_vector; 
            z_biased.row_mut(r_idx).copy_from(&row_sum);
        }
        self.z_cache = z_biased;
        
        self.activation_fn.activate(&self.z_cache)
    }

    pub fn backward(&mut self, gradient_wrt_z: &DMatrix<f32>, learning_rate: f32) -> DMatrix<f32> {
        assert_eq!(gradient_wrt_z.ncols(), self.weights.ncols(), "BACKWARD: Gradient_wrt_Z columns ({}) must match weights columns ({}) (output_size).", gradient_wrt_z.ncols(), self.weights.ncols());
        assert_eq!(gradient_wrt_z.nrows(), self.input_cache.nrows(), "BACKWARD: Gradient_wrt_Z rows ({}) must match batch size of cached input ({}).", gradient_wrt_z.nrows(), self.input_cache.nrows());

        let batch_size = self.input_cache.nrows() as f32;
        if batch_size == 0.0 { 
            // Return gradient for previous layer's activation, shape (0, prev_layer_output_size)
            // prev_layer_output_size is self.weights.nrows() (input_size to this layer)
            return DMatrix::zeros(0, self.weights.nrows()); 
        }


        // Calculate gradients for weights: dW = (1/m) * X_prev.T * dZ
        let dw = (&self.input_cache.transpose() * gradient_wrt_z) / batch_size;

        // ---- MANUAL CALCULATION FOR BIAS GRADIENTS (db_col_vector) ----
        let output_size_for_bias = self.biases.nrows(); // Number of neurons in this layer
        let mut calculated_db_col_vector_data = Vec::with_capacity(output_size_for_bias);

        for j in 0..output_size_for_bias { // For each output neuron / bias term
            // gradient_wrt_z.column(j) gives a ColumnVectorSlice (batch_size, 1)
            // .sum() on this slice sums its elements (sum over the batch for neuron j)
            let col_j_sum: f32 = gradient_wrt_z.column(j).sum(); 
            calculated_db_col_vector_data.push(col_j_sum / batch_size);
        }

        // Create a DVector (column vector) of shape (output_size_for_bias, 1)
        let db_col_vector = DVector::from_vec(calculated_db_col_vector_data);
        
        // ---- END OF MANUAL CALCULATION ----

        // Calculate gradient to pass to the previous layer: dError/dA_prev_layer = dZ * W.T
        let gradient_to_pass_back = gradient_wrt_z * self.weights.transpose();
        
        // Check shapes right before the problematic operation
        let bias_update_term = learning_rate * db_col_vector.clone(); // Clone for debug print if needed, use original for op

        // Update weights and biases
        self.weights -= learning_rate * dw;
        self.biases -= bias_update_term; // Using the pre-calculated term

        
        gradient_to_pass_back
    }
}