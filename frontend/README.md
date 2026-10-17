# Cybercrime ATM Zone Predictor — Frontend (Vite + React + Tailwind)

Web dashboard for the **Predictive Analytics Framework for Cybercrime Complaints** (SKIT Group 66).

## Overview
- **Complaint Submission**: Files cybercrime complaints with district, fraud type, amount, and time.
- **Backend Persistence**: Dispatches complaint records to the FastAPI service (`POST /complaints/`).
- **Zone Forecasting & Visualization**: Renders top-3 predicted cash-withdrawal ATM zones in Jaipur on an interactive Leaflet map.
- **Complaints Record**: Summarizes all filed complaints and top predicted withdrawal hotspots.

## Getting Started

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start development server:
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

4. Build for production:
   ```bash
   npm run build
   ```
