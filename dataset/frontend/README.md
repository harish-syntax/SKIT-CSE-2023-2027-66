# ATM Fraud Tracker — Form 2 UI (complaint form, prediction, result)

Flow: complaint form -> prediction screen (top 3 ATM zones) -> result screen (all complaints).

## Setup (inside a Vite + React project with Tailwind)

    npm install leaflet react-leaflet

Copy `tailwind.config.js` and everything in `src/` into your project.

## Connect the model

`src/api/predict.js` is a mock. Replace its body with the fetch call shown in the
comment once the FastAPI `/predict` endpoint is ready. Expected response:

    [{ id, name, lat, lng, probability }, ...3 items]
