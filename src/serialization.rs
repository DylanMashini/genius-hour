use serde::{Serialize, Deserialize};
use nalgebra::{DMatrix, DVector};
use crate::activation::ActivationFunction; // Your existing ActivationFunction
use crate::layer::DenseLayer;
use crate::network::NeuralNetwork;
use crate::loss::LossFunction; // Assuming LossFunction might be part of network state too

// --- Serializable representation of a DenseLayer ---
#[derive(Serialize, Deserialize, Debug)]
pub struct SerializableDenseLayer {
    weights_data: Vec<f32>,
    weights_rows: usize,
    weights_cols: usize,
    biases_data: Vec<f32>, // DVector is a column vector, so its length is its only dimension
    activation_fn: ActivationFunction,
}

impl From<&DenseLayer> for SerializableDenseLayer {
    fn from(layer: &DenseLayer) -> Self {
        Self {
            weights_data: layer.weights.as_slice().to_vec(), // Get flat data
            weights_rows: layer.weights.nrows(),
            weights_cols: layer.weights.ncols(),
            biases_data: layer.biases.as_slice().to_vec(), // Get flat data
            activation_fn: layer.activation_fn,
        }
    }
}

impl SerializableDenseLayer {
    // Converts back to a DenseLayer.
    // Note: input_size and output_size are derived from weights_cols and weights_rows
    // but DenseLayer::new also takes them. This is fine.
    pub fn into_dense_layer(self) -> DenseLayer {
        let mut layer = DenseLayer::new(self.weights_rows, self.weights_cols, self.activation_fn);
        layer.weights = DMatrix::from_vec(self.weights_rows, self.weights_cols, self.weights_data);
        layer.biases = DVector::from_vec(self.biases_data);
        layer
    }
}

// --- Serializable representation of a NeuralNetwork ---
#[derive(Serialize, Deserialize, Debug)]
pub struct SerializableNeuralNetwork {
    layers: Vec<SerializableDenseLayer>,
    // We might want to save loss_fn if it affects network behavior beyond training,
    // or if network reconstruction needs it. For now, assume it's set at creation.
    // loss_fn: SomeSerializableLossFunction, 
}

impl From<&NeuralNetwork> for SerializableNeuralNetwork {
    fn from(network: &NeuralNetwork) -> Self {
        let serializable_layers = network.get_layers().iter().map(SerializableDenseLayer::from).collect();
        Self {
            layers: serializable_layers,
        }
    }
}

impl SerializableNeuralNetwork {
    pub fn into_neural_network(self, loss_fn: LossFunction) -> NeuralNetwork {
        let mut nn = NeuralNetwork::new(loss_fn);
        for serializable_layer in self.layers {
            nn.add_layer(serializable_layer.into_dense_layer());
        }
        nn
    }
}

// Add a helper method to NeuralNetwork to get its layers for serialization
// This requires modifying the NeuralNetwork struct or its impl block.
// Let's assume we add this to network.rs:
/*
// In network.rs, within impl NeuralNetwork:
pub fn get_layers(&self) -> &Vec<DenseLayer> {
    &self.layers
}
*/