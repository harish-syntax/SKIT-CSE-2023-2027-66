import { useState } from "react";
import { atmZones } from "../data/zones";

export default function AlertScreen({ complaints = [], onSelectComplaint, onNavigate }) {
  const [filterPriority, setFilterPriority] = useState("ALL"); // ALL | CRITICAL | HIGH

  // Identify high-priority complaints that require immediate patrol pre-positioning
  const activeAlerts = complaints.filter((c) => {
    const isCritical = Number(c.amount) >= 50000 || (c.zones?.[0]?.probability || 0) >= 0.75;
    if (filterPriority === "CRITICAL") return isCritical;
    return true;
  });

  return (
    <div className="max-w-5xl space-y-6">
      {/* Top Header & Surveillance Status */}
      <div className="bg-white border border-surface-border rounded-lg p-5 shadow-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="font-display text-2xl font-bold text-ink">Police Patrol Alerts</h1>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-risk-50 text-risk border border-risk-100 animate-pulse">
              <span className="w-2 h-2 rounded-full bg-risk inline-block"></span>
              Live Monitoring Active
            </span>
          </div>
          <p className="text-sm text-muted mt-1">
            Real-time forecasted ATM withdrawal zones requiring pre-positioned patrol units across Jaipur.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigate("form")}
            className="bg-ink text-white text-xs font-medium px-4 py-2 rounded hover:bg-ink-800 transition-colors"
          >
            + File New Incident
          </button>
        </div>
      </div>

      {/* KPI Overview Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-surface-panel border border-surface-border rounded-lg p-4 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-muted uppercase tracking-wider">Active Alerts</span>
            <span className="text-base">🚨</span>
          </div>
          <p className="font-mono text-2xl font-bold text-ink mt-2">{activeAlerts.length}</p>
          <p className="text-xs text-muted mt-1">Requiring surveillance & patrol</p>
        </div>

        <div className="bg-surface-panel border border-surface-border rounded-lg p-4 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-muted uppercase tracking-wider">Primary ATM Cluster</span>
            <span className="text-base">📍</span>
          </div>
          <p className="font-mono text-xl font-bold text-risk mt-2">Mansarovar / Vaishali</p>
          <p className="text-xs text-muted mt-1">West & NW Residential Belt</p>
        </div>

        <div className="bg-surface-panel border border-surface-border rounded-lg p-4 shadow-xs">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-muted uppercase tracking-wider">Intervention Window</span>
            <span className="text-base">⏱️</span>
          </div>
          <p className="font-mono text-2xl font-bold text-safe mt-2">45–90 min</p>
          <p className="text-xs text-muted mt-1">Estimated pre-withdrawal window</p>
        </div>
      </div>

      {/* Main Content Layout: Alert Feed & Map Container Scaffold */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left Column: Alerts Feed */}
        <div className="lg:col-span-3 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-ink uppercase tracking-wider">Priority Incident Feed</h2>
            <div className="flex items-center gap-1.5 text-xs">
              <button
                onClick={() => setFilterPriority("ALL")}
                className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                  filterPriority === "ALL"
                    ? "bg-ink text-white"
                    : "bg-white text-muted border border-surface-border hover:text-ink"
                }`}
              >
                All Alerts
              </button>
              <button
                onClick={() => setFilterPriority("CRITICAL")}
                className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                  filterPriority === "CRITICAL"
                    ? "bg-risk text-white"
                    : "bg-white text-muted border border-surface-border hover:text-risk"
                }`}
              >
                Critical Only
              </button>
            </div>
          </div>

          {activeAlerts.length === 0 ? (
            <div className="bg-surface-panel border border-surface-border rounded-lg p-8 text-center">
              <p className="text-sm font-medium text-ink">No active patrol alerts right now</p>
              <p className="text-xs text-muted mt-1">
                New complaints filed via the complaint form will automatically populate high-risk patrol alerts here.
              </p>
              <button
                onClick={() => onNavigate("form")}
                className="mt-4 bg-ink text-white text-xs font-medium px-4 py-2 rounded hover:bg-ink-800 transition"
              >
                File a Complaint
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {activeAlerts.map((c) => {
                const topZone = c.zones?.[0];
                const prob = topZone ? Math.round(topZone.probability * 100) : 80;
                const isHigh = prob >= 70 || Number(c.amount) >= 50000;

                return (
                  <div
                    key={c.id}
                    onClick={() => onSelectComplaint && onSelectComplaint(c)}
                    className="bg-white border border-surface-border hover:border-steel rounded-lg p-4 transition-all shadow-xs cursor-pointer hover:shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-bold text-steel">{c.id}</span>
                          <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-surface text-ink-600 border border-surface-border">
                            {c.fraudType}
                          </span>
                          {isHigh && (
                            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-risk-50 text-risk border border-risk-100">
                              HIGH THREAT
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-muted mt-1">
                          Victim: <span className="font-medium text-ink">{c.victim || "Anonymous"}</span> ({c.victim_district || "Jaipur"})
                        </p>
                      </div>

                      <div className="text-right">
                        <span className="font-mono text-sm font-bold text-ink">
                          ₹{Number(c.amount).toLocaleString("en-IN")}
                        </span>
                        <div className="text-[11px] text-muted">Loss Reported</div>
                      </div>
                    </div>

                    {/* Predicted Patrol Target */}
                    <div className="mt-3.5 pt-3 border-t border-surface-border flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs">🎯 Target ATM Zone:</span>
                        <span className="text-xs font-bold text-ink">
                          {topZone?.name || "Mansarovar Cluster"}
                        </span>
                        <span className="font-mono text-xs font-bold text-risk bg-risk-50 px-2 py-0.5 rounded">
                          {prob}% Conf.
                        </span>
                      </div>

                      <span className="text-[11px] font-semibold text-steel hover:underline flex items-center gap-1">
                        View Map →
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right Column: Zone Readiness & Map Scaffold Preview */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white border border-surface-border rounded-lg p-4 shadow-xs">
            <h3 className="text-xs font-bold text-ink uppercase tracking-wider mb-3">
              Monitored Jaipur ATM Zones (10)
            </h3>
            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {atmZones.map((z, idx) => (
                <div
                  key={z.id}
                  className="flex items-center justify-between p-2 rounded bg-surface hover:bg-ink-50/60 transition text-xs"
                >
                  <div>
                    <span className="font-mono text-[11px] text-muted mr-1.5">#{idx + 1}</span>
                    <span className="font-medium text-ink">{z.name}</span>
                  </div>
                  <span className="text-[10px] font-mono text-muted">{z.region.split(" ")[0]}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-ink text-ink-50 rounded-lg p-4 shadow-xs">
            <div className="flex items-center gap-2">
              <span className="text-base">🛡️</span>
              <h4 className="text-xs font-bold uppercase tracking-wider">Patrol Pre-positioning Rule</h4>
            </div>
            <p className="text-xs text-ink-100 mt-2 leading-relaxed">
              Upon receiving a complaint, dispatch police beat officers to the forecasted ATM cluster within 45 minutes to intercept cash drainage attempts.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
