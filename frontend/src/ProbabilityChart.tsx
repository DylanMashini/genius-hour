import React from 'react';
// import './ProbabilityChart.css'; // Removed import

interface ProbabilityChartProps {
  probabilities: number[] | Float32Array;
  predictedDigit: number | null; // While parent logic implies non-null, keeping it flexible.
}

const ProbabilityChart: React.FC<ProbabilityChartProps> = ({ probabilities, predictedDigit }) => {
  // If data is missing or malformed, display an informative message.
  // The wrapper ensures consistent styling with the main chart view.
  if (!probabilities || probabilities.length !== 10) {
    return (
      <div className="bg-[#1a1a1a] p-6 rounded-lg mt-6 border border-[#444]">
        <p className="text-[#b0b0b0] text-center italic py-4">
          No probability data available or data is malformed.
        </p>
      </div>
    );
  }

  // Convert Float32Array to a standard array for easier manipulation if needed,
  // though .map() works directly on Float32Array.
  const probsArray = Array.from(probabilities);

  return (
    <div className="bg-[#1a1a1a] p-6 rounded-lg mt-6 border border-[#444]">
      <h3 className="text-[#b0b0b0] mt-0 mb-5 text-lg font-semibold text-center">Prediction Probabilities</h3>
      <ul className="list-none p-0 m-0">
        {probsArray.map((prob, index) => {
          // Ensure probability is between 0 and 1, then convert to percentage.
          // Clamping helps guard against potential unexpected values from the model.
          const clampedProb = Math.max(0, Math.min(1, prob));
          const percentage = clampedProb * 100;
          const displayPercentage = percentage.toFixed(1); // e.g., "95.3"
          const isPredicted = index === predictedDigit;

          return (
            <li
              key={index}
              className="flex items-center gap-3 mb-[0.7rem] last:mb-0"
              aria-label={`Digit ${index}: ${displayPercentage}% probability`}
            >
              <span className="text-[#e0e0e0] font-bold text-base w-[2ch] text-center flex-shrink-0">{index}</span>
              <div className="flex-grow h-6 bg-[#2c2c2c] rounded-[5px] overflow-hidden border border-[#444]">
                <div
                  className={`h-full rounded-[3px] transition-all duration-300 ease-in-out ${isPredicted ? 'bg-[#4caf50]' : 'bg-[#61dafb]'}`}
                  style={{ width: `${percentage}%` }} // Use percentage for bar width
                  role="progressbar"
                  aria-valuenow={percentage} // Numeric value for accessibility
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  {/* Bar is purely visual; text is displayed next to it */}
                </div>
              </div>
              <span className="text-[#b0b0b0] tabular-nums w-[60px] text-right text-sm flex-shrink-0">{displayPercentage}%</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
};

export default ProbabilityChart;