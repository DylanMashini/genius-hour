import { useState, useEffect, useRef, useCallback } from 'react';
// import './App.css'; // Removed App.css import
import init, { predict_mnist } from '../../genius-hour/pkg/'; // Adjust path if needed
import ProbabilityChart from './ProbabilityChart';
import Writeup from './Writeup';

const MNIST_SIZE = 28;
const DISPLAY_CANVAS_SIZE = 280; // Visual size of the canvas
const PIXEL_SCALE = DISPLAY_CANVAS_SIZE / MNIST_SIZE; // How large each "pixel" of the 28x28 grid is on display

const initialGrid = () => new Array(MNIST_SIZE * MNIST_SIZE).fill(0);

function App() {
  const [isWasmReady, setIsWasmReady] = useState(false);
  const [result, setResult] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [probabilities, setProbabilities] = useState<number[] | Float32Array | null>(null);


  // imageData now stores the 28x28 grid (0 for black, 1 for white)
  const [gridData, setGridData] = useState<number[]>(initialGrid());

  const canvasRef = useRef<HTMLCanvasElement>(null);

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

  // Function to draw the grid onto the canvas
  const drawGrid = useCallback(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous frame

    for (let y = 0; y < MNIST_SIZE; y++) {
      for (let x = 0; x < MNIST_SIZE; x++) {
        const index = y * MNIST_SIZE + x;
        ctx.fillStyle = gridData[index] === 1 ? 'white' : 'black';
        ctx.fillRect(x * PIXEL_SCALE, y * PIXEL_SCALE, PIXEL_SCALE, PIXEL_SCALE);
      }
    }
    // Optional: Add grid lines for better visual separation of pixels
    ctx.strokeStyle = '#333'; // Dark gray grid lines
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= MNIST_SIZE; i++) {
      ctx.beginPath();
      ctx.moveTo(i * PIXEL_SCALE, 0);
      ctx.lineTo(i * PIXEL_SCALE, DISPLAY_CANVAS_SIZE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * PIXEL_SCALE);
      ctx.lineTo(DISPLAY_CANVAS_SIZE, i * PIXEL_SCALE);
      ctx.stroke();
    }

  }, [gridData]);

  // Effect to draw the grid whenever gridData changes
  useEffect(() => {
    drawGrid();
  }, [gridData, drawGrid]);

  const getGridCoordinates = (event: React.MouseEvent<HTMLCanvasElement>): { gridX: number; gridY: number } | null => {
    if (!canvasRef.current) return null;
    const rect = canvasRef.current.getBoundingClientRect();
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;

    const gridX = Math.floor(canvasX / PIXEL_SCALE);
    const gridY = Math.floor(canvasY / PIXEL_SCALE);

    if (gridX >= 0 && gridX < MNIST_SIZE && gridY >= 0 && gridY < MNIST_SIZE) {
      return { gridX, gridY };
    }
    return null;
  };

  const handlePixelInteraction = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getGridCoordinates(event);
    if (coords) {
      const { gridX, gridY } = coords;
      const index = gridY * MNIST_SIZE + gridX;

      // Create a new array to trigger re-render
      setGridData(prevGridData => {
        const newGridData = [...prevGridData];
        // Toggle pixel: if you want to erase, you'd need a different logic or modifier key
        // For now, drawing always sets to white (1)
        newGridData[index] = 1;
        return newGridData;
      });
    }
  };

  const startDrawing = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    handlePixelInteraction(event); // Draw the first pixel
  }, []); // handlePixelInteraction dependency will be managed by its own definition

  const draw = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    handlePixelInteraction(event);
  }, [isDrawing]); // handlePixelInteraction dependency

  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const clearCanvas = useCallback(() => {
    setGridData(initialGrid());
    setResult(null);
    setProbabilities(null); // Clear probabilities too
  }, []);


  const runInference = async () => {
    if (!isWasmReady) {
      console.warn("WASM not ready");
      return;
    }
    setIsLoading(true);
    setResult(null);
    setProbabilities(null);

    // The gridData is already 0s and 1s.
    // MNIST typically expects 0.0 for black and 1.0 for white.
    // Our gridData: 0 for black, 1 for white. So this matches.
    const processedData = new Float32Array(gridData);

    // console.log("Processed Data Sample (first 10):", Array.from(processedData.slice(0,10)));

    try {
      const predictionResult = predict_mnist(processedData);
      console.log("Raw prediction:", predictionResult);
      setProbabilities(predictionResult);
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
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="font-sans text-gray-200 w-full max-w-xl mx-auto text-center bg-neutral-800 p-8 rounded-xl shadow-2xl flex flex-col gap-6 my-8">
      <header>
        <h1 className="text-[#61dafb] mb-2 text-4xl">Genius Hour Demo</h1>
        <p className="text-neutral-400 text-lg mt-0 mb-4">Draw a digit between 0 and 9, and the model will try to predict what digit you drew. </p>
      </header>

      <main className="flex flex-col gap-6">
        <div className="flex justify-center items-center border-2 border-neutral-700 rounded-lg bg-black overflow-hidden w-fit mx-auto">
          <canvas
            ref={canvasRef}
            width={DISPLAY_CANVAS_SIZE}
            height={DISPLAY_CANVAS_SIZE}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing} // Stop drawing if mouse leaves canvas
            style={{ touchAction: 'none' }}
            className="block cursor-crosshair rounded-md"
          />
        </div>

        <div className="flex justify-center gap-4 mt-4">
          <button
            onClick={runInference}
            disabled={!isWasmReady}
            className="!bg-[#61dafb] text-[#1a1a1a] border-none py-[0.8rem] px-6 rounded-md text-base font-bold cursor-pointer transition-colors duration-200 ease-in-out ease hover:enabled:bg-[#4cb8d8] hover:enabled:-translate-y-0.5 active:enabled:translate-y-0 disabled:bg-[#555555] disabled:text-[#888888] disabled:cursor-not-allowed"
          >
            {isLoading ? 'Predicting...' : (isWasmReady ? 'Run Inference' : 'Loading WASM...')}
          </button>
          <button
            onClick={clearCanvas}
            disabled={isLoading}
            className="text-white border-none py-[0.8rem] px-6 rounded-md text-base font-bold cursor-pointer transition-colors duration-200 ease-in-out ease hover:enabled:bg-[#d33c23] hover:enabled:-translate-y-0.5 active:enabled:translate-y-0 disabled:bg-[#555555] disabled:text-[#888888] disabled:cursor-not-allowed"
          >
            Clear Canvas
          </button>
        </div>

        {result !== null && (
          <div className="mt-6 p-4 bg-neutral-900 rounded-lg border border-neutral-700">
            <h2 className="m-0 text-2xl text-neutral-400">Predicted Digit: <span className="text-5xl font-bold text-[#4caf50] inline-block ml-2">{result}</span></h2>
          </div>
        )}
        {/* Render the ProbabilityChart */}
        {probabilities && result !== null && (
          <ProbabilityChart probabilities={probabilities} predictedDigit={result} />
        )}

        {!isWasmReady && <p className="text-neutral-400 italic">WASM module is loading, please wait...</p>}
      </main>

      <Writeup />
    </div>
  );
}

export default App;