import { useState } from "react";
import Sidebar from "./components/Sidebar";
import ComplaintForm from "./components/ComplaintForm";
import PredictionScreen from "./components/PredictionScreen";
import ResultScreen from "./components/ResultScreen";
import { predictTopZones } from "./api/predict";

export default function App() {
  const [screen, setScreen] = useState("form"); // form | prediction | results
  const [complaints, setComplaints] = useState([]);
  const [current, setCurrent] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(data) {
    setSubmitting(true);
    try {
      const zones = await predictTopZones(data);
      const complaint = {
        ...data,
        id: `CC-2026-${Math.floor(10000 + Math.random() * 89999)}`,
        filedAt: new Date().toISOString(),
        zones,
      };
      setComplaints((list) => [complaint, ...list]);
      setCurrent(complaint);
      setScreen("prediction");
    } finally {
      setSubmitting(false);
    }
  }

  const navKey = screen === "results" ? "results" : "form";

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
          />
        )}
      </main>
    </div>
  );
}
