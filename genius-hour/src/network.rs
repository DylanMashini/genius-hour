use nalgebra::DMatrix;
use crate::layer::DenseLayer;
use crate::loss::LossFunction;
use crate::activation::ActivationFunction;
use crate::serialization::SerializableNeuralNetwork;
use std::fs::File;
use std::io::{BufWriter, BufReader};
use bincode::{serialize_into, deserialize_from};

pub struct NeuralNetwork {
    layers: Vec<DenseLayer>,
    loss_fn: LossFunction,
}

impl NeuralNetwork {
    pub fn new(loss_fn: LossFunction) -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            loss_fn,
        }
    }

    pub fn get_layers(&self) -> &Vec<DenseLayer> {
        &self.layers
    }

    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: &DMatrix<f32>) -> DMatrix<f32> {
        let mut current_output = input.clone();
        for layer in self.layers.iter_mut() {
            // Corrected line: pass by reference ¤t_output
            current_output = layer.forward(&current_output); 
        }
        current_output
    }

    pub fn train_batch(
        &mut self, 
        inputs: &DMatrix<f32>, 
        targets: &DMatrix<f32>, 
        learning_rate: f32
    ) -> f32 {
        // Forward pass
        // This also caches inputs and z_values in layers, to avoid recalculation
        let predictions = self.predict(inputs); 

        // Calculate loss
        let loss = self.loss_fn.calculate(&predictions, targets);

        // Backward pass
        // Calculate initial gradient: dError/dZ_L for the last layer L
        let mut d_error_dz: DMatrix<f32>;
        let last_layer_idx = self.layers.len() - 1;

        // Special case for Softmax + CrossEntropy: dLoss/dZ = Predictions - Targets
        if self.layers[last_layer_idx].activation_fn == ActivationFunction::Softmax &&
           self.loss_fn == LossFunction::CrossEntropy {
            let batch_size = predictions.nrows() as f32;
            if batch_size == 0.0 { return loss; } // Avoid division by zero if batch is empty
            d_error_dz = (&predictions - targets) / batch_size;
        } else {
            // General case: dError/dZ_L = dError/dA_L * dA_L/dZ_L
            let d_error_da = self.loss_fn.derivative(&predictions, targets); 
            // Bad: Cloning z_cache every time is expensive, but I couldn't get a mutable reference to it
            let last_layer_z_cache = self.layers[last_layer_idx].z_cache.clone(); 
            let da_dz = self.layers[last_layer_idx].activation_fn.derivative(&last_layer_z_cache); 
            d_error_dz = d_error_da.component_mul(&da_dz); 
        }

        // Propagate gradient backwards starting from the last layer
        let mut gradient_from_next_layer_wrt_activation = 
            self.layers[last_layer_idx].backward(&d_error_dz, learning_rate);

        // For hidden layers (from L-1 down to 0)
        for i in (0..last_layer_idx).rev() {
            // gradient_from_next_layer_wrt_activation is dError/dA_current
            // Accessing z_cache, clone to avoid borrow conflicts.
            let current_layer_z_cache = self.layers[i].z_cache.clone(); 
            let da_dz_current = self.layers[i].activation_fn.derivative(&current_layer_z_cache);
            
            d_error_dz = gradient_from_next_layer_wrt_activation.component_mul(&da_dz_current);
            
            gradient_from_next_layer_wrt_activation = 
                self.layers[i].backward(&d_error_dz, learning_rate);
        }
        loss
    }

    pub fn save_weights(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serializable_nn = SerializableNeuralNetwork::from(self);
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serialize_into(writer, &serializable_nn)?;
        Ok(())
    }

    pub fn load_weights(path: &str, loss_fn: LossFunction) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let serializable_nn: SerializableNeuralNetwork = deserialize_from(reader)?;
        Ok(serializable_nn.into_neural_network(loss_fn))
    }


}