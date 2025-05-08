// neural_network_lib/src/lib.rs

// Modules of your library
pub mod activation;
pub mod layer;
pub mod loss;
pub mod network;
pub mod serialization; // Assuming this contains SerializableNeuralNetwork etc.

// Re-export key structs/enums for easier use within the crate or by other Rust crates
pub use activation::ActivationFunction;
pub use layer::DenseLayer;
pub use loss::LossFunction;
pub use network::NeuralNetwork;

#[cfg(target_arch = "wasm32")]
mod wasm_specific {
    use super::*; // Make parent module items available
    use nalgebra::DMatrix;
    use once_cell::sync::Lazy;
    use wasm_bindgen::prelude::*; // For global static initialized once

    // Helper to set panic hook for better WASM debugging (optional)
    #[cfg(feature = "dev")]
    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
        // You can add other init logic here if needed when WASM module loads
    }

    // Path to your trained model file (relative to Cargo.toml of this crate)
    const MODEL_BYTES: &[u8] = include_bytes!("../mnist_model.bincode"); // Adjust path if model is elsewhere

    // Global static for the loaded neural network.
    // Lazy ensures it's loaded only once when first accessed.
    static MNIST_NETWORK: Lazy<Result<NeuralNetwork, String>> = Lazy::new(|| {
        // Deserialize the network from the embedded bytes.
        // We need to specify the LossFunction used during training.
        // For MNIST with softmax output, it was CrossEntropy.
        match bincode::deserialize(MODEL_BYTES) {
            Ok(serializable_nn) => {
                let snn: super::serialization::SerializableNeuralNetwork = serializable_nn; // Corrected path
                Ok(snn.into_neural_network(LossFunction::CrossEntropy))
            }
            Err(e) => Err(format!("Failed to deserialize embedded model: {}", e)),
        }
    });

    // WASM function to initialize and check the model (optional, mostly for testing)
    #[wasm_bindgen]
    pub fn ensure_model_loaded() -> Result<(), JsValue> {
        match &*MNIST_NETWORK {
            // Access the Lazy static
            Ok(_) => Ok(()),
            Err(s) => Err(JsValue::from_str(&format!("Model loading failed: {}", s))),
        }
    }

    // WASM function to perform prediction.
    // Input: a flat slice of f32 representing a single flattened image (e.g., 784 pixels).
    // Output: a Vec<f32> representing the probabilities for each class (e.g., 10 probabilities).
    #[wasm_bindgen]
    pub fn predict_mnist(image_data: &[f32]) -> Result<Vec<f32>, JsValue> {
        // 1. Validate input length
        let expected_input_size = 28 * 28; // MNIST image size
        if image_data.len() != expected_input_size {
            return Err(JsValue::from_str(&format!(
                "Invalid input image data length. Expected {}, got {}",
                expected_input_size,
                image_data.len()
            )));
        }

        use std::cell::RefCell;

        thread_local! {
            static THREAD_LOCAL_NETWORK: RefCell<Result<NeuralNetwork, String>> = RefCell::new(
                match bincode::deserialize(MODEL_BYTES) {
                    Ok(serializable_nn) => {
                        let snn: super::serialization::SerializableNeuralNetwork = serializable_nn; // Corrected path
                        Ok(snn.into_neural_network(LossFunction::CrossEntropy))
                    }
                    Err(e) => Err(format!("Failed to deserialize embedded model: {}", e)),
                }
            );
        }

        THREAD_LOCAL_NETWORK.with(|network_cell| {
            match *network_cell.borrow_mut() {
                Ok(ref mut nn) => {
                    let input_matrix = DMatrix::from_row_slice(1, expected_input_size, image_data);
                    let output_matrix = nn.predict(&input_matrix);
                    Ok(output_matrix.as_slice().to_vec())
                }
                Err(ref s) => Err(JsValue::from_str(&format!(
                    "Model not loaded or error: {}",
                    s
                ))),
            }
        })
    }

    #[wasm_bindgen]
    pub fn get_model_input_size() -> usize {
        28 * 28
    }

    #[wasm_bindgen]
    pub fn get_model_output_size() -> usize {
        match &*MNIST_NETWORK {
            Ok(nn) => {
                if let Some(last_layer) = nn.get_layers().last() {
                    last_layer.biases.nrows()
                } else {
                    0
                }
            }
            Err(_) => 0,
        }
    }
}

// Optional: Re-export WASM specific functions if you want them available at crate::function_name
// when wasm32 is targeted. Otherwise, they'd be at crate::wasm_specific::function_name
#[cfg(target_arch = "wasm32")]
pub use wasm_specific::*;
