import { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import ComplaintForm from "./components/ComplaintForm";
import PredictionScreen from "./components/PredictionScreen";
import ResultScreen from "./components/ResultScreen";
import AlertScreen from "./components/AlertScreen";
import { predictTopZones } from "./api/predict";
import { createComplaint, listComplaints } from "./api/complaints";

const STORAGE_KEY = "skit_cybercrime_complaints_v1";

// Safely loads previously filed complaints from browser localStorage
function getInitialComplaints() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) {
        return parsed;
      }
    }
  } catch (e) {
    console.warn("Error reading complaints from localStorage:", e);
  }
  return [];
}

export default function App() {
  const [screen, setScreen] = useState("form"); // form | prediction | results | alerts
  const [complaints, setComplaints] = useState(getInitialComplaints);
  const [current, setCurrent] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // Sync with backend API if available, preserving and merging with local stored cases
  useEffect(() => {
    async function loadData() {
      try {
        const backendComplaints = await listComplaints();
        if (backendComplaints && backendComplaints.length > 0) {
          setComplaints((prev) => {
            const existingIds = new Set(prev.map((c) => c.id));
            const newFromBackend = backendComplaints
              .filter((bc) => !existingIds.has(bc.incident_id || `CC-${bc.id}`))
              .map((bc) => ({
                id: bc.incident_id || `CC-${bc.id}`,
                victim: bc.victim || `Victim (${bc.victim_district})`,
                victim_district: bc.victim_district,
                fraudType: bc.fraud_type,
                amount: bc.amount_inr,
                filedAt: bc.created_at || new Date().toISOString(),
                zones: bc.target_atm_zone
                  ? [{ name: bc.target_atm_zone, probability: 0.85, lat: 26.8523, lng: 75.7667 }]
                  : [],
              }));

            if (newFromBackend.length === 0) return prev;
            const merged = [...newFromBackend, ...prev];
            try {
              localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
            } catch (err) {
              console.warn("Error caching merged complaints:", err);
            }
            return merged;
          });
        }
      } catch (err) {
        console.warn("Backend not reachable on initial load, using persistent local storage:", err.message);
      }
    }
    loadData();
  }, []);

  async function handleSubmit(data) {
    setSubmitting(true);
    try {
      // 1. Get predicted ATM zones using predictive model
      const zones = await predictTopZones(data);
      const topZone = zones[0]?.name || "Mansarovar";

      // 2. Persist to FastAPI Backend (/complaints/)
      const backendPayload = {
        incident_id: data.incident_id || `INC-${Date.now()}`,
        victim_district: data.victim_district || "Jaipur",
        fraud_type: data.fraud_type || data.fraudType,
        amount_inr: data.amount_inr || data.amount,
        time_of_complaint: data.time_of_complaint || "12:00",
        time_to_withdraw: "01:30",
        target_atm_zone: topZone,
        specific_atm_location: `${topZone} Main Road Cluster`,
      };

      const saved = await createComplaint(backendPayload);

      // 3. Update local state and persist to localStorage
      const newComplaint = {
        id: saved.incident_id || data.incident_id,
        victim: data.victim,
        contact: data.contact,
        victim_district: data.victim_district,
        fraudType: data.fraud_type || data.fraudType,
        amount: data.amount_inr || data.amount,
        filedAt: saved.created_at || new Date().toISOString(),
        zones,
      };

      setComplaints((list) => {
        const updated = [newComplaint, ...list];
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        } catch (e) {
          console.warn("Could not save complaint to localStorage:", e);
        }
        return updated;
      });

      setCurrent(newComplaint);
      setScreen("prediction");
    } catch (err) {
      console.error("Submission error:", err);
    } finally {
      setSubmitting(false);
    }
  }

  function handleClearComplaints() {
    if (window.confirm("Are you sure you want to clear all complaints history?")) {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch (e) {
        console.warn("Error removing from localStorage:", e);
      }
      setComplaints([]);
    }
  }

  const navKey = screen === "results" ? "results" : screen === "alerts" ? "alerts" : "form";

  return (
    <div className="min-h-screen flex bg-surface">
      <Sidebar active={navKey} onNavigate={setScreen} />
      <main className="flex-1 p-8 overflow-auto">
        {screen === "form" && (
          <ComplaintForm onSubmit={handleSubmit} submitting={submitting} />
        )}
        {screen === "prediction" && current && (
          <PredictionScreen
            complaint={current}
            zones={current.zones}
            onViewAll={() => setScreen("results")}
            onNew={() => setScreen("form")}
          />
        )}
        {screen === "results" && (
          <ResultScreen
            complaints={complaints}
            onSelect={(c) => {
              setCurrent(c);
              setScreen("prediction");
            }}
            onNew={() => setScreen("form")}
            onClear={handleClearComplaints}
          />
        )}
        {screen === "alerts" && (
          <AlertScreen
            complaints={complaints}
            onSelectComplaint={(c) => {
              setCurrent(c);
              setScreen("prediction");
            }}
            onNavigate={setScreen}
          />
        )}
      </main>
    </div>
  );
}
