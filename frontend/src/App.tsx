import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import init, { predict_mnist } from '../../genius-hour/pkg/'; // Adjust path if needed

const CANVAS_SIZE = 280; // Display canvas size (e.g., 10x MNIST resolution)
const MNIST_SIZE = 28;   // MNIST image size

function App() {
  const [isWasmReady, setIsWasmReady] = useState(false);
  const [result, setResult] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isLoading, setIsLoading] = useState(false); // For inference loading state

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastPositionRef = useRef<{ x: number; y: number } | null>(null);

  // Initialize WASM
  useEffect(() => {
    console.log("Initializing WASM...");
    init()
      .then(() => {
        console.log("WASM Initialized");
        setIsWasmReady(true);
      })
      .catch(err => {
        console.error("Failed to initialize WASM:", err);
      });
  }, []);

  // Initialize canvas
  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'white';
        // Make lines thicker for better visibility when downscaled
        ctx.lineWidth = CANVAS_SIZE / MNIST_SIZE * 1.5; // Adjust thickness as needed
      }
    }
  }, []); // Runs once on mount

  const getCanvasCoordinates = (event: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } | null => {
    if (!canvasRef.current) return null;
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  };

  const startDrawing = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoordinates(event);
    if (coords) {
      setIsDrawing(true);
      lastPositionRef.current = coords;
      // Draw a dot for single clicks
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) {
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
        ctx.lineTo(coords.x, coords.y); // Draw a tiny line to make a dot
        ctx.stroke();
      }
    }
  }, []);

  const draw = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !lastPositionRef.current || !canvasRef.current) return;
    const coords = getCanvasCoordinates(event);
    if (coords) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.beginPath();
        ctx.moveTo(lastPositionRef.current.x, lastPositionRef.current.y);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
        lastPositionRef.current = coords;
      }
    }
  }, [isDrawing]);

  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
    lastPositionRef.current = null;
  }, []);

  const clearCanvas = useCallback(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        setResult(null);
      }
    }
  }, []);

  const runInference = async () => {
    if (!isWasmReady || !canvasRef.current) {
      console.warn("WASM not ready or canvas not found");
      return;
    }
    setIsLoading(true);
    setResult(null);

    const mainCanvas = canvasRef.current;
    
    // 1. Create a temporary 28x28 canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = MNIST_SIZE;
    tempCanvas.height = MNIST_SIZE;
    const tempCtx = tempCanvas.getContext('2d');

    if (!tempCtx) {
      console.error("Failed to get 2D context for temp canvas");
      setIsLoading(false);
      return;
    }

    // 2. Draw the content of the main canvas onto the temporary canvas, scaling it down
    // This performs downsampling.
    tempCtx.drawImage(mainCanvas, 0, 0, mainCanvas.width, mainCanvas.height, 0, 0, MNIST_SIZE, MNIST_SIZE);

    // 3. Get image data from the temporary 28x28 canvas
    const imageData = tempCtx.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE);
    const pixelData = imageData.data; // Uint8ClampedArray: R,G,B,A, R,G,B,A, ...

    // 4. Convert to grayscale Float32Array (0.0 - 1.0)
    // MNIST expects white digit on black background (0=black, 1=white)
    // Our canvas is white on black. getImageData gives [R,G,B,A].
    // We can take any of R, G, or B as they should be the same for grayscale.
    // Normalize to 0-1.
    const processedData = new Float32Array(MNIST_SIZE * MNIST_SIZE);
    for (let i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
      processedData[i] = pixelData[i * 4] / 255.0; // Using Red channel
    }
    
    // console.log("Processed Data Sample (first 10):", Array.from(processedData.slice(0,10)));

    try {
      // The `predict_mnist` from your original snippet suggests it takes Float32Array
      // and returns an array of probabilities.
      const predictionResult = predict_mnist(processedData);
      // console.log("Raw prediction:", predictionResult);

      // Find the index of the max value (predicted digit)
      let maxIndex = 0;
      let maxValue = predictionResult[0];
      for (let i = 1; i < predictionResult.length; i++) {
        if (predictionResult[i] > maxValue) {
          maxValue = predictionResult[i];
          maxIndex = i;
        }
      }
      setResult(maxIndex);
    } catch (error) {
      console.error("Error during prediction:", error);
      setResult(null); // Or set an error state
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>MNIST Digit Recognizer</h1>
        <p className="subtitle">Draw a digit (0-9) below and see the prediction!</p>
      </header>

      <main>
        <div className="canvas-container">
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseOut={stopDrawing} // Stop drawing if mouse leaves canvas while pressed
            style={{ touchAction: 'none' }} // Good for touch devices
          />
        </div>

        <div className="controls">
          <button onClick={runInference} disabled={!isWasmReady || isLoading}>
            {isLoading ? 'Predicting...' : (isWasmReady ? 'Run Inference' : 'Loading WASM...')}
          </button>
          <button onClick={clearCanvas} disabled={isLoading}>Clear Canvas</button>
        </div>

        {result !== null && (
          <div className="result-display">
            <h2>Predicted Digit: <span className="prediction">{result}</span></h2>
          </div>
        )}
        {!isWasmReady && <p className="status-text">WASM module is loading, please wait...</p>}
      </main>

      <footer>
        <p>Powered by Rust (WASM) & React</p>
      </footer>
    </div>
  );
}

export default App;