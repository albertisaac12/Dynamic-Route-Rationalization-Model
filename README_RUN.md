# Dynamic Route Rationalization - Frontend & Backend

This project now includes a Flask backend for serving the model and a React frontend for user interaction.

## Prerequisites
- Python 3.8+
- Node.js & npm

## Setup & Run

### 1. Start the Backend
The backend serves the model inference API.

1. Open a terminal in `C:\Dynaim Route Rationalization\Dynamic-Route-Rationalization-Model`.
2. Install Flask (if not already installed):
   ```bash
   pip install flask
   ```
3. Run the server:
   ```bash
   python Backend/app.py
   ```
   The server will start on `http://localhost:5000`.

### 2. Start the Frontend
The frontend provides the user interface.

1. Open a new terminal in `C:\Dynaim Route Rationalization\Dynamic-Route-Rationalization-Model\Frontend`.
2. Install dependencies (if you haven't already):
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open the URL shown in the terminal (usually `http://localhost:5173`) in your browser.

## Usage
- Select Source and Destination from the dropdowns.
- Enter weather and traffic conditions.
- Click "Predict Optimal Route" to see the result.
