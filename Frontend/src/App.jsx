import React, { useState, useEffect } from 'react';
import axios from 'axios';

const LOCATIONS = [
  "Hyderabad", "Gachibowli", "Madhapur", "Banjara Hills", "Secunderabad",
  "Begumpet", "Kukatpally", "Uppal", "LB Nagar", "Dilsukhnagar",
  "Hitech City", "Kondapur", "Financial District", "Airport",
  "Jubilee Hills", "BHEL", "KPHB"
];

// Mock coordinates for simulation (x, y in percentage)
const LOCATION_COORDS = {
  "Hyderabad": { x: 50, y: 50 },
  "Gachibowli": { x: 20, y: 60 },
  "Madhapur": { x: 30, y: 55 },
  "Banjara Hills": { x: 45, y: 45 },
  "Secunderabad": { x: 60, y: 30 },
  "Begumpet": { x: 55, y: 40 },
  "Kukatpally": { x: 25, y: 35 },
  "Uppal": { x: 80, y: 40 },
  "LB Nagar": { x: 75, y: 65 },
  "Dilsukhnagar": { x: 70, y: 60 },
  "Hitech City": { x: 25, y: 50 },
  "Kondapur": { x: 25, y: 45 },
  "Financial District": { x: 15, y: 65 },
  "Airport": { x: 50, y: 90 },
  "Jubilee Hills": { x: 40, y: 50 },
  "BHEL": { x: 10, y: 30 },
  "KPHB": { x: 20, y: 30 }
};

function MapSimulation({ source, destination, via, isAnimating }) {
  const start = LOCATION_COORDS[source] || { x: 50, y: 50 };
  const end = LOCATION_COORDS[destination] || { x: 50, y: 50 };
  const viaPoint = via ? (LOCATION_COORDS[via] || { x: 50, y: 50 }) : null;

  return (
    <div className="relative w-full h-64 bg-slate-900 rounded-xl border border-slate-700 overflow-hidden shadow-inner group">
      {/* Grid Background */}
      <div className="absolute inset-0 opacity-20"
        style={{ backgroundImage: 'linear-gradient(#334155 1px, transparent 1px), linear-gradient(90deg, #334155 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
      </div>

      {/* Radar Scan Effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/10 to-transparent w-full h-full animate-[scan_4s_linear_infinite] opacity-30"></div>

      {/* Connection Lines */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        {/* Input Route (Dashed) */}
        <line
          x1={`${start.x}%`} y1={`${start.y}%`}
          x2={`${end.x}%`} y2={`${end.y}%`}
          stroke="#06b6d4"
          strokeWidth="2"
          strokeDasharray="5,5"
          className="opacity-20"
        />

        {/* Predicted Route (Solid, Glowing) - Source -> Via -> Destination */}
        {viaPoint && (
          <>
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7" />
              </marker>
            </defs>

            {/* Leg 1: Source -> Via */}
            <line
              x1={`${start.x}%`} y1={`${start.y}%`}
              x2={`${viaPoint.x}%`} y2={`${viaPoint.y}%`}
              stroke="#a855f7"
              strokeWidth="3"
              strokeLinecap="round"
              className="animate-pulse shadow-[0_0_10px_#a855f7]"
            />
            <line
              x1={`${start.x}%`} y1={`${start.y}%`}
              x2={`${viaPoint.x}%`} y2={`${viaPoint.y}%`}
              stroke="#a855f7"
              strokeWidth="3"
              markerEnd="url(#arrowhead)"
              className="opacity-80"
            />

            {/* Leg 2: Via -> Destination */}
            <line
              x1={`${viaPoint.x}%`} y1={`${viaPoint.y}%`}
              x2={`${end.x}%`} y2={`${end.y}%`}
              stroke="#a855f7"
              strokeWidth="3"
              strokeLinecap="round"
              className="animate-pulse shadow-[0_0_10px_#a855f7]"
            />
            <line
              x1={`${viaPoint.x}%`} y1={`${viaPoint.y}%`}
              x2={`${end.x}%`} y2={`${end.y}%`}
              stroke="#a855f7"
              strokeWidth="3"
              markerEnd="url(#arrowhead)"
              className="opacity-80"
            />
          </>
        )}

        {isAnimating && (
          <circle r="4" fill="#22d3ee">
            <animateMotion
              dur="2s"
              repeatCount="indefinite"
              path={`M${start.x * 4} ${start.y * 2.5} L${end.x * 4} ${end.y * 2.5}`}
            />
          </circle>
        )}
      </svg>

      {/* Points */}
      {Object.entries(LOCATION_COORDS).map(([name, coords]) => {
        const isSource = name === source;
        const isDest = name === destination;
        const isVia = name === via;

        let pointClass = 'bg-slate-600 hover:bg-slate-400 w-2 h-2';

        if (isVia) {
          pointClass = 'bg-purple-500 w-5 h-5 shadow-[0_0_20px_#a855f7] z-30 animate-bounce';
        } else if (isSource) {
          pointClass = 'bg-green-500 w-4 h-4 shadow-[0_0_15px_#22c55e] z-20';
        } else if (isDest) {
          pointClass = 'bg-red-500 w-4 h-4 shadow-[0_0_15px_#ef4444] z-20';
        }

        return (
          <div
            key={name}
            className={`absolute rounded-full transform -translate-x-1/2 -translate-y-1/2 transition-all duration-300 ${pointClass}`}
            style={{ left: `${coords.x}%`, top: `${coords.y}%` }}
            title={name}
          >
            {(isSource || isDest || isVia) && (
              <span className={`absolute -top-8 left-1/2 transform -translate-x-1/2 text-xs font-bold px-2 py-1 rounded border whitespace-nowrap z-40
                ${isVia ? 'bg-purple-900/80 text-purple-100 border-purple-500' : 'bg-slate-800 text-white border-slate-600'}`}>
                {name}
                {isVia && <span className="ml-1 text-[10px] opacity-75">(Via)</span>}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function App() {
  const [formData, setFormData] = useState({
    source: 'Hyderabad',
    destination: 'Gachibowli',
    temp: 28,
    tempmax: 30,
    tempmin: 22,
    humidity: 55,
    windspeed: 3,
    conditions: 0,
    traffic_distance_m: 9400,
    traffic_duration_min: 22,
    traffic_pressure: 3.2,
    traffic_efficiency: 510,
    is_weekend: 0,
    day_of_week: 3,
    hour_of_day: 18
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'source' || name === 'destination' ? value : Number(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('/predict', formData);
      // Simulate a small delay for the animation to be appreciated
      await new Promise(resolve => setTimeout(resolve, 1500));
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-800 to-gray-900 text-white p-8 font-sans">
      <div className="max-w-6xl mx-auto">
        <header className="mb-10 text-center">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 mb-2">
            Dynamic Route Rationalization
          </h1>
          <p className="text-slate-400">AI-Powered Route Prediction Model</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Form Section */}
          <div className="lg:col-span-2 space-y-8">
            <div className="bg-white/5 backdrop-blur-lg border border-white/10 rounded-2xl p-6 shadow-xl">
              <h2 className="text-xl font-semibold mb-6 text-cyan-300 border-b border-white/10 pb-2">Input Parameters</h2>
              <form onSubmit={handleSubmit} className="space-y-6">

                {/* Locations */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm text-slate-300">Source</label>
                    <select
                      name="source"
                      value={formData.source}
                      onChange={handleChange}
                      className="w-full bg-slate-800/50 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500 transition-all"
                    >
                      {LOCATIONS.map(loc => <option key={loc} value={loc}>{loc}</option>)}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm text-slate-300">Destination</label>
                    <select
                      name="destination"
                      value={formData.destination}
                      onChange={handleChange}
                      className="w-full bg-slate-800/50 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500 transition-all"
                    >
                      {LOCATIONS.map(loc => <option key={loc} value={loc}>{loc}</option>)}
                    </select>
                  </div>
                </div>

                {/* Weather Data */}
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Weather Conditions</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {[
                      { label: 'Temp (°C)', name: 'temp' },
                      { label: 'Max Temp', name: 'tempmax' },
                      { label: 'Min Temp', name: 'tempmin' },
                      { label: 'Humidity (%)', name: 'humidity' },
                      { label: 'Windspeed', name: 'windspeed' },
                      { label: 'Conditions (Code)', name: 'conditions' },
                    ].map(field => (
                      <div key={field.name} className="space-y-1">
                        <label className="text-xs text-slate-500">{field.label}</label>
                        <input
                          type="number"
                          name={field.name}
                          value={formData[field.name]}
                          onChange={handleChange}
                          className="w-full bg-slate-800/50 border border-slate-700 rounded px-3 py-1.5 focus:border-cyan-500 focus:outline-none transition-colors"
                        />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Traffic Data */}
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Traffic & Time</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {[
                      { label: 'Distance (m)', name: 'traffic_distance_m' },
                      { label: 'Duration (min)', name: 'traffic_duration_min' },
                      { label: 'Pressure', name: 'traffic_pressure' },
                      { label: 'Efficiency', name: 'traffic_efficiency' },
                      { label: 'Hour (0-23)', name: 'hour_of_day' },
                      { label: 'Day (0-6)', name: 'day_of_week' },
                    ].map(field => (
                      <div key={field.name} className="space-y-1">
                        <label className="text-xs text-slate-500">{field.label}</label>
                        <input
                          type="number"
                          name={field.name}
                          value={formData[field.name]}
                          onChange={handleChange}
                          className="w-full bg-slate-800/50 border border-slate-700 rounded px-3 py-1.5 focus:border-cyan-500 focus:outline-none transition-colors"
                        />
                      </div>
                    ))}
                    <div className="flex items-center space-x-2 pt-6">
                      <input
                        type="checkbox"
                        name="is_weekend"
                        checked={formData.is_weekend === 1}
                        onChange={(e) => setFormData(prev => ({ ...prev, is_weekend: e.target.checked ? 1 : 0 }))}
                        className="w-4 h-4 text-cyan-500 rounded focus:ring-cyan-500 bg-slate-800 border-slate-600"
                      />
                      <label className="text-sm text-slate-300">Is Weekend?</label>
                    </div>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold py-3 px-6 rounded-lg shadow-lg transform transition-all hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Analyzing Routes...' : 'Predict Optimal Route'}
                </button>
              </form>
            </div>
          </div>

          {/* Right Column: Map & Results */}
          <div className="lg:col-span-1 space-y-6">

            {/* Map Simulation */}
            <div className="bg-white/5 backdrop-blur-lg border border-white/10 rounded-2xl p-4 shadow-xl">
              <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">Live Route Simulation</h3>
              <MapSimulation
                source={formData.source}
                destination={formData.destination}
                via={result ? result.destination : null}
                isAnimating={loading}
              />
            </div>

            {/* Result Card */}
            <div className="bg-white/5 backdrop-blur-lg border border-white/10 rounded-2xl p-6 shadow-xl flex flex-col min-h-[300px]">
              <h2 className="text-xl font-semibold mb-4 text-purple-300 border-b border-white/10 pb-2">Prediction Result</h2>

              {loading && (
                <div className="flex-1 flex flex-col items-center justify-center space-y-4 animate-pulse">
                  <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
                  <p className="text-cyan-400">Optimizing path...</p>
                </div>
              )}

              {error && (
                <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200">
                  <strong>Error:</strong> {error}
                </div>
              )}

              {result && !loading && (
                <div className="space-y-6 animate-fade-in">
                  <div className="text-center p-6 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-xl border border-cyan-500/30">
                    <p className="text-sm text-cyan-300 uppercase tracking-widest mb-1">Optimal Route ID</p>
                    <p className="text-6xl font-bold text-white tracking-tighter">{result.predicted_route_id}</p>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                      <span className="text-slate-400 text-sm">From</span>
                      <span className="font-medium text-cyan-100">{formData.source}</span>
                    </div>
                    <div className="flex justify-center text-slate-500">
                      ↓
                    </div>
                    <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                      <span className="text-slate-400 text-sm">To</span>
                      <span className="font-medium text-cyan-100">{formData.destination}</span>
                    </div>
                  </div>

                  <div className="mt-auto pt-4 text-center">
                    <p className="text-xs text-slate-500">Confidence: High (Simulated)</p>
                  </div>
                </div>
              )}

              {!result && !loading && !error && (
                <div className="flex-1 flex items-center justify-center text-slate-500 text-center p-4">
                  <p>Enter parameters and click predict to see the best route.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
