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

// WASM library caused problems when trying to compile to train, so conditionally exclude it
#[cfg(target_arch = "wasm32")]
mod wasm_specific {
    use super::*;
    use nalgebra::DMatrix;
    use once_cell::sync::Lazy;
    use wasm_bindgen::prelude::*;

    // Debug MOde 
    #[cfg(feature = "dev")]
    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
    }

    // include_bytes! is a compile time macro that includes the contents of a file as a byte slice.
    const MODEL_BYTES: &[u8] = include_bytes!("../mnist_model.bincode"); // Adjust path if model is elsewhere

    // Lazy static for the loaded neural network, only loaded when needed, still available in global scope and never double loaded
    static MNIST_NETWORK: Lazy<Result<NeuralNetwork, String>> = Lazy::new(|| {
        // We need to specify the LossFunction used during training.
        match bincode::deserialize(MODEL_BYTES) {
            Ok(serializable_nn) => {
                let snn: super::serialization::SerializableNeuralNetwork = serializable_nn;
                Ok(snn.into_neural_network(LossFunction::CrossEntropy))
            }
            Err(e) => Err(format!("Failed to deserialize embedded model: {}", e)),
        }
    });

    // WASM function to perform prediction.
    // Input: a Float32Array representing a single flattened image (e.g., 784 pixels).
    // Output: a Float32Array representing the probabilities for each class (e.g., 10 probabilities).
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

        // thread_local! keeps us memory safe while preventing reloading the model
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

}

// Re-export WASM specific functions, as they were placed in their own module
#[cfg(target_arch = "wasm32")]
pub use wasm_specific::*;
