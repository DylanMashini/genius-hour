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

// --- WASM Specific Code ---
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
            let snn: serialization::SerializableNeuralNetwork = serializable_nn;
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

    // 2. Access the loaded network
    // We need a mutable reference to call `predict` because `predict` takes `&mut self`
    // due to internal caching in layers during forward pass.
    // If MNIST_NETWORK was just `NeuralNetwork` not `Result`, we'd need a Mutex for interior mutability.
    // For inference-only, `predict` could ideally take `&self` if caches are not strictly needed
    // or handled differently for inference.
    //
    // WORKAROUND for `&mut self` predict with Lazy<Result<NeuralNetwork, _>>:
    // This is tricky. Lazy gives &T. If T is NeuralNetwork, we can't get &mut T.
    // Option A: Make predict take `&self` if possible (ideal for inference).
    // Option B: Clone the network on each call (inefficient).
    // Option C: Use a thread_local static Mutex<NeuralNetwork> (more complex).
    //
    // Let's assume we modify `NeuralNetwork::predict` to take `&self` for this WASM case
    // OR we modify how caches are handled in `DenseLayer::forward` for an inference mode.
    // For now, to make it work without altering existing library code too much,
    // we might have to re-deserialize or clone if `predict` absolutely needs `&mut self`.
    // This is a significant design consideration for WASM.
    //
    // Simplest "hack" if predict MUST be &mut self and we can't change it:
    // Re-deserialize on each call (VERY INEFFICIENT, just for demonstration)
    /*
    let mut nn = match bincode::deserialize(MODEL_BYTES) {
        Ok(snn_data) => {
            let snn: serialization::SerializableNeuralNetwork = snn_data;
            snn.into_neural_network(LossFunction::CrossEntropy)
        },
        Err(e) => return Err(JsValue::from_str(&format!("Failed to deserialize model for prediction: {}", e))),
    };
    */

    // Better approach if `predict` must be `&mut self`: Use `RefCell` or `Mutex` inside `Lazy`.
    // For simplicity, let's assume `predict` is actually `&self` or we'll face issues.
    // If `DenseLayer::forward` modifies `self.input_cache` and `self.z_cache`, then `predict` must be `&mut self`.
    // This is a problem for concurrent calls in WASM if it's truly shared state.
    // If single-threaded WASM, we could use `RefCell` around `NeuralNetwork` in the `Lazy`.

    // Let's proceed AS IF predict could take &self by temporarily creating a new instance
    // from the deserialized form for each call. This is NOT performant for real use
    // but shows the data flow. A production system would use Mutex/RefCell or an
    // inference-specific forward pass.

    // To avoid re-deserializing every time, and if we can't change `predict` to `&self`,
    // and we are in a single-threaded WASM environment (typical for browser main thread):
    // We can use a thread_local static RefCell.

    use std::cell::RefCell;

    thread_local! {
        // Each "thread" (WASM instance in this context) gets its own network instance.
        // This avoids issues with &mut self on a shared static.
        static THREAD_LOCAL_NETWORK: RefCell<Result<NeuralNetwork, String>> = RefCell::new(
            match bincode::deserialize(MODEL_BYTES) {
                Ok(serializable_nn) => {
                    let snn: serialization::SerializableNeuralNetwork = serializable_nn;
                    Ok(snn.into_neural_network(LossFunction::CrossEntropy))
                }
                Err(e) => Err(format!("Failed to deserialize embedded model: {}", e)),
            }
        );
    }

    THREAD_LOCAL_NETWORK.with(|network_cell| {
        match *network_cell.borrow_mut() {
            // Get mutable access
            Ok(ref mut nn) => {
                // 3. Convert input slice to DMatrix
                // Input is a single image, so batch size is 1.
                // DMatrix::from_row_slice expects &Vec or slice.
                let input_matrix = DMatrix::from_row_slice(1, expected_input_size, image_data);

                // 4. Perform prediction
                let output_matrix = nn.predict(&input_matrix); // This is &mut self

                // 5. Convert output DMatrix to Vec<f32>
                // Output matrix will be (1, num_classes). We want its data as a flat Vec.
                Ok(output_matrix.as_slice().to_vec())
            }
            Err(ref s) => Err(JsValue::from_str(&format!(
                "Model not loaded or error: {}",
                s
            ))),
        }
    })
}

// Optional: A function to get model metadata (e.g., input/output sizes)
#[wasm_bindgen]
pub fn get_model_input_size() -> usize {
    28 * 28
}

#[wasm_bindgen]
pub fn get_model_output_size() -> usize {
    // This is harder if the network isn't loaded yet or if it failed.
    // We can try to access the static network or hardcode it if known.
    // For a robust solution, ensure_model_loaded should be called first.
    match &*MNIST_NETWORK {
        Ok(nn) => {
            if let Some(last_layer) = nn.get_layers().last() {
                last_layer.biases.nrows() // Output size is number of biases in last layer
            } else {
                0 // Or some error indicator
            }
        }
        Err(_) => 0, // Or some error indicator
    }
}
