use serde::{Serialize, Deserialize};
use nalgebra::{DMatrix, DVector};
use crate::activation::ActivationFunction; // Your existing ActivationFunction
use crate::layer::DenseLayer;
use crate::network::NeuralNetwork;
use crate::loss::LossFunction; // Assuming LossFunction might be part of network state too

#[derive(Serialize, Deserialize, Debug)]
pub struct SerializableDenseLayer {
    weights_data: Vec<f32>,
    weights_rows: usize,
    weights_cols: usize,
    biases_data: Vec<f32>,
    activation_fn: ActivationFunction,
}

impl From<&DenseLayer> for SerializableDenseLayer {
    fn from(layer: &DenseLayer) -> Self {
        Self {
            weights_data: layer.weights.as_slice().to_vec(),
            weights_rows: layer.weights.nrows(),
            weights_cols: layer.weights.ncols(),
            biases_data: layer.biases.as_slice().to_vec(),
            activation_fn: layer.activation_fn,
        }
    }
}

impl SerializableDenseLayer {
    // Converts back to a DenseLayer.
    pub fn into_dense_layer(self) -> DenseLayer {
        let mut layer = DenseLayer::new(self.weights_rows, self.weights_cols, self.activation_fn);
        layer.weights = DMatrix::from_vec(self.weights_rows, self.weights_cols, self.weights_data);
        layer.biases = DVector::from_vec(self.biases_data);
        layer
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SerializableNeuralNetwork {
    layers: Vec<SerializableDenseLayer>,
    // TODO: Serialize loss_fn, and metadata like training date
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