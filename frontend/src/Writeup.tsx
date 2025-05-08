export default function Writeup() {
  return (
    <div>
      <h1 className="text-2xl font-bold">Genius Hour Project</h1>
      <h2 className="text-xl font-bold">Overview</h2>
      <p className="text-left w-full pl-8">For my genius hour project, I built a simple neural network from scratch in Rust, only using a linear algebra library for the math. I then trained it on the MNIST dataset, and used it to classify handwritten digits. The source code is available on <a href="https://github.com/DylanMashini/genius-hour" target="_blank" rel="noopener noreferrer">GitHub</a>.</p>
      <h2 className="text-xl font-bold">Implementation</h2>
      <p className="text-left w-full pl-8">The neural network struct is implemented in <Code>src/network.rs</Code>. It is flexible enough to support any number of layers and neurons, but for the handwriting recognition task I only used a single hidden layer. </p>

      <h2 className="text-xl font-bold mt-4">Codebase Structure</h2>
      <p className="text-left w-full pl-8">The project is organized into several Rust modules, each handling a specific part of the neural network:</p>
      <ul className="list-disc list-inside text-left w-full pl-12 space-y-1">
        <li><Code>src/main.rs</Code>: In Rust, main.rs is a special file that is the entry point of the program. It contains the main application logic, including loading the data, initializing the network, running the training loop, and evaluating the model. It orchestrates the overall execution flow. I would like to move some of the execution logic into the <Code>NeuralNetwork</Code> struct, so you could call a single train function and have everything handled for you. </li>
        <li><Code>src/network.rs</Code>: Defines the <Code>NeuralNetwork</Code> struct, which encapsulates the collection of layers and the chosen loss function. It implements the forward pass (prediction) and the backpropagation algorithm (<Code>train_batch</Code>) for updating weights.</li>
        <li><Code>src/layer.rs</Code>: Implements the <Code>DenseLayer</Code> struct, representing a fully connected layer in the network. This includes weight and bias initialization, the forward propagation step for a single layer (dot product of the input and weights, then adding the bias), and the backward pass to compute gradients for its weights and biases.</li>
        <li><Code>src/activation.rs</Code>: Provides the <Code>ActivationFunction</Code> enum (e.g., ReLU, Sigmoid, Softmax, Linear) and their respective activation and derivative functions, crucial for both forward and backward passes.</li>
        <li><Code>src/loss.rs</Code>: Defines the <Code>LossFunction</Code> enum (e.g., Mean Squared Error, Cross-Entropy) and methods to calculate the loss and its derivative with respect to the network's output.</li>
        <li><Code>src/serialization.rs</Code>: Handles the saving and loading of the trained neural network model. It defines serializable versions of the network and layer structs to persist weights and biases, allowing for model reuse without retraining. This was very important, because it allowed me to pretrain a model on my computer, then load it on the demo site. This prevents the need to train the model from scratch every time the page is loaded, which would be very computationally expensive.</li>
        <li><Code>src/lib.rs</Code>: This file serves as the library's main entry point for WebAssembly compilation. WebAssembly is a binary format that can run in the browser, allowing me to have a browser-based demo that doesn't need a server. When compiling the library for WASM, it includes the pretrained MNIST model in the output binary, using Rust's <Code>include_bytes!</Code> macro. The main export of this file is the <Code>predict_mnist</Code> function. This function can be called from JavaScript to perform inference in the browser, enabling the demo. </li>
      </ul>

      <h2 className="text-xl font-bold mt-4">Key Features</h2>
      <ul className="list-disc list-inside text-left w-full pl-12 space-y-1">
        <li><strong>Modular Design:</strong> The codebase is well-structured into modules like <Code>network</Code>, <Code>layer</Code>, <Code>activation</Code>, and <Code>loss</Code>, promoting clarity and maintainability.</li>
        <li><strong>Customizable Network Architecture:</strong> The <Code>NeuralNetwork</Code> struct supports an arbitrary number of <Code>DenseLayer</Code> instances, each with a configurable number of neurons and activation functions.</li>
        <li><strong>Backpropagation from Scratch:</strong> The training process, including the forward and backward passes (backpropagation), is implemented manually, demonstrating a fundamental understanding of neural network mechanics.</li>
        <li><strong>Multiple Activation and Loss Functions:</strong> The network supports various activation functions (ReLU, Sigmoid, Softmax, Linear) and loss functions (Cross-Entropy, Mean Squared Error), allowing flexibility for different types of problems.</li>
        <li><strong>Model Persistence:</strong> Implemented functionality to save trained model weights to a file and load them later, which is essential for reusing models without retraining. This is handled in <Code>serialization.rs</Code>.</li>
        <li><strong>MNIST Dataset Handling:</strong> Includes utilities (in <Code>mnist_loader.rs</Code>, referenced by <Code>main.rs</Code>) for loading and preprocessing the MNIST dataset for training and testing.</li>
        <li><strong>Batch Training:</strong> Batch training is implemented, but has very poor performance for some reason. I couldn't figure out why, so I ended up training without batching. If I can figure out the issue, batching will greatly improve the speed of training the model. </li>
      </ul>

      <h2 className="text-xl font-bold mt-4">Improvements</h2>
      <p className="text-left w-full pl-8">Currently, the performance of the demo is ok, but it sometimes makes mistakes, even when the number looks very clear. One significant improvement would be to implement a more advanced optimization algorithm, such as the Adam optimizer. Right now, we use basic stochastic gradient descent, which is essentially what we learned in class for multivariable optimization problems. Adam can lead to better model accuracy. When training a neural net, the learning rate influences the speed we go "down the gradient". If the learning rate is too high, the model will continue to overshoot the minimum (loss), and if it is too low, it may take too long to converge and find the minimum. Adam can help with this by adapting the learning rate for each parameter, rather than globally. </p>

      <h2 className="text-xl font-bold mt-4">Other Considerations</h2>
      <p className="text-left w-full pl-8">When I was trying to figure out why the model wasn't performing as well as I would've liked it to in the demo, I realized that there is a subtle difference between the task I trained the model for, and the task the model is expected to perform. The MNIST dataset is a collection of 28x28 pixel images of handwritten digits in greyscale (0-255). The demo however, is a collection of 28x28 pixel images of digits in black and white. This means that in the training data, each pixel has different intensities, but in the demo every pixel is either 0 or 255. This slight difference in data could contribute to the model's confusion. </p>
    </div>

  );
}

const Code = ({children}: {children: React.ReactNode}) => {
  return (
    <code className="bg-neutral-900 text-neutral-400 px-1 rounded-md">
      {children}
    </code>
  );
};