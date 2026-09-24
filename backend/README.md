# Backend — Cybercrime Complaint & Prediction API

Covers sprint rows:
1. **Setup project and database** — FastAPI app initialized, SQLite database + tables created on startup.
2. **Build Complaint API** — `Complaint` and `Prediction` tables, full CRUD API for complaints.

## Run it

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs.

## Endpoints

| Method | Path                | Description                     |
|--------|---------------------|----------------------------------|
| GET    | `/`                  | Health check                    |
| POST   | `/complaints/`       | Create a complaint              |
| GET    | `/complaints/`       | List complaints (filter/paginate) |
| GET    | `/complaints/{id}`   | Get one complaint               |
| PATCH  | `/complaints/{id}`   | Update a complaint              |
| DELETE | `/complaints/{id}`   | Delete a complaint              |

## Fields (match `dataset/jaipur_cybercrime_5000_detailed.csv`)

`incident_id`, `victim_district`, `fraud_type`, `amount_inr`, `time_of_complaint`,
`time_to_withdraw`, `target_atm_zone`, `specific_atm_location`.

## Next sprint rows (not built yet)

- Connect ML model to backend
- Build Prediction API (serve predictions from the trained model)
- Build History API
- API integration testing
- Security & authentication
