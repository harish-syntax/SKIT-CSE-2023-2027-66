const API_BASE = "http://127.0.0.1:8000";

/**
 * Creates a new complaint in the backend FastAPI service.
 * Falls back gracefully with warning if backend service is not running.
 */
export async function createComplaint(payload) {
  try {
    const res = await fetch(`${API_BASE}/complaints/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    console.warn("Backend API not reachable or error:", err.message);
    // Return mock-compatible object so frontend continues smoothly
    return {
      id: Math.floor(1000 + Math.random() * 9000),
      incident_id: payload.incident_id,
      victim_district: payload.victim_district,
      fraud_type: payload.fraud_type,
      amount_inr: payload.amount_inr,
      time_of_complaint: payload.time_of_complaint,
      created_at: new Date().toISOString(),
      offline: true,
    };
  }
}

/**
 * Fetches all complaints from the FastAPI backend.
 */
export async function listComplaints(filters = {}) {
  try {
    const params = new URLSearchParams();
    if (filters.victim_district) params.append("victim_district", filters.victim_district);
    if (filters.fraud_type) params.append("fraud_type", filters.fraud_type);

    const url = `${API_BASE}/complaints/${params.toString() ? "?" + params.toString() : ""}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("Could not fetch complaints from backend:", err.message);
    return [];
  }
}
