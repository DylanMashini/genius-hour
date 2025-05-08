use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::io::{stdout, Write}; // For flushing print output

// Use components from the current crate
use genius_hour::activation::ActivationFunction;
use genius_hour::layer::DenseLayer;
use genius_hour::loss::LossFunction;
use genius_hour::network::NeuralNetwork;


// Declare the mnist_loader module, which should be in src/mnist_loader.rs
mod mnist_loader;

const IMAGE_FEATURE_SIZE: usize = 28 * 28;
const NUM_CLASSES: usize = 10;

fn calculate_accuracy(
    network: &mut NeuralNetwork,
    test_images: &DMatrix<f32>,
    test_labels_raw: &DMatrix<f32>, // Raw labels (0-9), not one-hot
) -> f32 {
    if test_images.nrows() == 0 {
        return 0.0;
    }
    let predictions = network.predict(test_images); // Forward pass for all test images
    let mut correct_predictions = 0;

    for i in 0..predictions.nrows() {
        let pred_row = predictions.row(i); // Probabilities for each class for image i
        let actual_label_val = test_labels_raw[(i, 0)]; // Get the scalar label

        // Find the class with the highest probability
        let (predicted_class, _max_prob) = pred_row.iter().enumerate().fold(
            (0, -1.0f32), // (index_of_max, max_value)
            |(idx_max, val_max), (idx, &val)| {
                if val > val_max {
                    (idx, val)
                } else {
                    (idx_max, val_max)
                }
            },
        );

        if predicted_class == actual_label_val as usize {
            correct_predictions += 1;
        }
    }
    correct_predictions as f32 / predictions.nrows() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "mnist_model.bincode"; // Path to save/load the model

    // --- Option 1: Train a new model or load if exists ---
    let mut nn: NeuralNetwork;
    if std::path::Path::new(model_path).exists() {
        println!("Loading existing model from {}...", model_path);
        // You need to know the original LossFunction to reconstruct, or save it too.
        // For MNIST, it's CrossEntropy.
        nn = NeuralNetwork::load_weights(model_path, LossFunction::CrossEntropy)?;
        println!("Model loaded.");

        // Optionally, evaluate the loaded model immediately
        // (Ensure test data is loaded if you do this here)
        // let test_images = mnist_loader::load_mnist_images("mnist/t10k-images.idx3-ubyte")?;
        // let test_labels_raw = mnist_loader::load_mnist_labels("mnist/t10k-labels.idx1-ubyte", false)?;
        // let accuracy = calculate_accuracy(&mut nn, &test_images, &test_labels_raw);
        // println!("Loaded model initial test accuracy: {:.2}%", accuracy * 100.0);

    } else {
        println!("No existing model found. Training a new one.");
        // --- Network Definition (if training new) ---
        nn = NeuralNetwork::new(LossFunction::CrossEntropy);
        nn.add_layer(DenseLayer::new(
            IMAGE_FEATURE_SIZE,
            128,
            ActivationFunction::ReLU,
        ));
        nn.add_layer(DenseLayer::new(128, 64, ActivationFunction::ReLU));
        nn.add_layer(DenseLayer::new(
            64,
            NUM_CLASSES,
            ActivationFunction::Softmax,
        ));

        // --- Load MNIST Data (only if training) ---
        println!("Loading MNIST data for training...");
        let train_images_path = "mnist/train-images.idx3-ubyte";
        let train_labels_path = "mnist/train-labels.idx1-ubyte";
        
        let train_images = mnist_loader::load_mnist_images(train_images_path)?;
        let train_labels_one_hot = mnist_loader::load_mnist_labels(train_labels_path, true)?;
        
        println!("Train images: {}x{}", train_images.nrows(), train_images.ncols());
        println!("Train labels (one-hot): {}x{}", train_labels_one_hot.nrows(), train_labels_one_hot.ncols());

        // --- Training Hyperparameters ---
        let epochs = 30; // Or fewer if just testing save/load
        let learning_rate = 0.01;
        let batch_size = 64;

        println!("\nStarting training...");
        println!("Epochs: {}, Learning Rate: {}, Batch Size: {}", epochs, learning_rate, batch_size);

        let num_samples = train_images.nrows();
        let mut indices: Vec<usize> = (0..num_samples).collect();
        let mut rng = thread_rng();

        for epoch in 0..epochs {
            indices.shuffle(&mut rng);
            let mut epoch_loss = 0.0;
            let mut num_batches_processed = 0;

            for batch_start in (0..num_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_samples);
                if batch_start >= batch_end { continue; }
                let current_batch_indices = &indices[batch_start..batch_end];
                let (batch_inputs, batch_targets) = mnist_loader::get_mini_batch(
                    &train_images, &train_labels_one_hot, current_batch_indices);
                if batch_inputs.nrows() > 0 {
                    let batch_loss = nn.train_batch(&batch_inputs, &batch_targets, learning_rate);
                    epoch_loss += batch_loss;
                    num_batches_processed += 1;
                }
                if num_batches_processed > 0 && num_batches_processed % 100 == 0 {
                    print!(".");
                    stdout().flush()?;
                }
            }
            println!(); 
            
            // Evaluation after each epoch (requires test data to be loaded)
            let test_images_path = "mnist/t10k-images.idx3-ubyte";
            let test_labels_path = "mnist/t10k-labels.idx1-ubyte";
            let test_images_eval = mnist_loader::load_mnist_images(test_images_path)?;
            let test_labels_raw_eval = mnist_loader::load_mnist_labels(test_labels_path, false)?;
            let accuracy = calculate_accuracy(&mut nn, &test_images_eval, &test_labels_raw_eval);
            
            let avg_epoch_loss = if num_batches_processed > 0 { epoch_loss / num_batches_processed as f32 } else { 0.0 };
            println!("Epoch {}/{} - Avg Loss: {:.6} - Test Accuracy: {:.2}%", epoch + 1, epochs, avg_epoch_loss, accuracy * 100.0);
        }
        println!("\nTraining finished.");
        // Save the trained model
        println!("Saving model to {}...", model_path);
        nn.save_weights(model_path)?;
        println!("Model saved.");
    }


    // --- Inference/Evaluation part (can run always, on new or loaded model) ---
    println!("\nEvaluating final model performance...");
    let test_images_path = "mnist/t10k-images.idx3-ubyte";
    let test_labels_path = "mnist/t10k-labels.idx1-ubyte";
    let test_images = mnist_loader::load_mnist_images(test_images_path)?;
    let test_labels_raw = mnist_loader::load_mnist_labels(test_labels_path, false)?;

    let final_accuracy = calculate_accuracy(&mut nn, &test_images, &test_labels_raw);
    println!("Final Test Accuracy on the model: {:.2}%", final_accuracy * 100.0);

    // Example of predicting a single image (or a small batch)
    if test_images.nrows() > 0 {
        let single_image_batch = test_images.rows(0, 1).clone_owned(); // Get the first image as a 1xN matrix
        let single_prediction = nn.predict(&single_image_batch);
        println!("Prediction for the first test image: {:?}", single_prediction);
        let actual_label = test_labels_raw[(0,0)];
        println!("Actual label for the first test image: {}", actual_label);
    }

    Ok(())
}
