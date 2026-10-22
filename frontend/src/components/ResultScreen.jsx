import { useState, useMemo } from "react";
import { fraudTypes } from "../data/zones";

function formatTime(iso) {
  if (!iso) return "N/A";
  try {
    return new Date(iso).toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function getPriorityBadge(amount) {
  const amt = Number(amount) || 0;
  if (amt >= 50000) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-risk-50 text-risk border border-risk-100">
        High Risk
      </span>
    );
  }
  if (amt >= 15000) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
        Medium
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-slate-50 text-slate-600 border border-slate-200">
      Standard
    </span>
  );
}

export default function ResultScreen({ complaints = [], onSelect, onNew }) {
  const [search, setSearch] = useState("");
  const [fraudFilter, setFraudFilter] = useState("ALL");
  const [priorityFilter, setPriorityFilter] = useState("ALL");

  // Filtered complaints based on search query and dropdown filters
  const filtered = useMemo(() => {
    return complaints.filter((c) => {
      const q = search.toLowerCase().trim();
      const matchSearch =
        !q ||
        (c.id && c.id.toLowerCase().includes(q)) ||
        (c.victim && c.victim.toLowerCase().includes(q)) ||
        (c.victim_district && c.victim_district.toLowerCase().includes(q)) ||
        (c.fraudType && c.fraudType.toLowerCase().includes(q));

      const matchFraud = fraudFilter === "ALL" || c.fraudType === fraudFilter;

      const amt = Number(c.amount) || 0;
      let matchPriority = true;
      if (priorityFilter === "HIGH") matchPriority = amt >= 50000;
      else if (priorityFilter === "MEDIUM") matchPriority = amt >= 15000 && amt < 50000;
      else if (priorityFilter === "STANDARD") matchPriority = amt < 15000;

      return matchSearch && matchFraud && matchPriority;
    });
  }, [complaints, search, fraudFilter, priorityFilter]);

  const totalAmount = useMemo(() => complaints.reduce((s, c) => s + (Number(c.amount) || 0), 0), [complaints]);
  const filteredAmount = useMemo(() => filtered.reduce((s, c) => s + (Number(c.amount) || 0), 0), [filtered]);

  // Export to CSV function for mentor documentation
  function handleExportCSV() {
    if (filtered.length === 0) return;
    const headers = ["Case_ID", "Victim", "District", "Fraud_Type", "Amount_INR", "Top_Predicted_Zone", "Probability", "Filed_At"];
    const rows = filtered.map((c) => [
      `"${c.id || ""}"`,
      `"${c.victim || "Unknown"}"`,
      `"${c.victim_district || "Jaipur"}"`,
      `"${c.fraudType || ""}"`,
      Number(c.amount) || 0,
      `"${c.zones?.[0]?.name || "N/A"}"`,
      c.zones?.[0]?.probability ? `${Math.round(c.zones[0].probability * 100)}%` : "N/A",
      `"${c.filedAt || ""}"`,
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `cybercrime_complaints_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  return (
    <div className="max-w-5xl">
      {/* Header section */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <div>
          <h1 className="font-display text-2xl text-ink font-semibold">Complaints Case Repository</h1>
          <p className="text-sm text-muted">
            All submitted cybercrime complaints with forecasted ATM cash-withdrawal hotspots.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            onClick={handleExportCSV}
            disabled={filtered.length === 0}
            className="border border-surface-border bg-white text-ink-600 text-xs font-medium px-3.5 py-2 rounded hover:border-ink-400 transition-colors disabled:opacity-50 flex items-center gap-1.5"
            title="Download complaints as CSV table"
          >
            <span>📥</span> Export CSV
          </button>
          <button
            onClick={onNew}
            className="bg-ink text-white text-xs font-medium px-4 py-2 rounded hover:bg-ink-800 transition-colors flex items-center gap-1.5"
          >
            <span>+</span> File New Case
          </button>
        </div>
      </div>

      {/* Metrics overview */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5 mb-6">
        <div className="bg-surface-panel border border-surface-border rounded-lg px-4 py-3 shadow-xs">
          <p className="text-xs font-medium text-muted uppercase tracking-wider">Total Complaints</p>
          <div className="flex items-baseline justify-between mt-1">
            <span className="font-mono text-2xl font-bold text-ink">{complaints.length}</span>
            <span className="text-xs text-muted">Active Session</span>
          </div>
        </div>

        <div className="bg-surface-panel border border-surface-border rounded-lg px-4 py-3 shadow-xs">
          <p className="text-xs font-medium text-muted uppercase tracking-wider">Total Loss Incurred</p>
          <div className="flex items-baseline justify-between mt-1">
            <span className="font-mono text-2xl font-bold text-ink">₹{totalAmount.toLocaleString("en-IN")}</span>
            <span className="text-xs text-risk font-medium">Reported</span>
          </div>
        </div>

        <div className="bg-surface-panel border border-surface-border rounded-lg px-4 py-3 shadow-xs">
          <p className="text-xs font-medium text-muted uppercase tracking-wider">Filtered Volume</p>
          <div className="flex items-baseline justify-between mt-1">
            <span className="font-mono text-2xl font-bold text-steel">{filtered.length} Cases</span>
            <span className="text-xs text-muted font-mono">₹{filteredAmount.toLocaleString("en-IN")}</span>
          </div>
        </div>
      </div>

      {/* Search and Filters Bar */}
      <div className="bg-surface-panel border border-surface-border rounded-t-lg p-3.5 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between">
        <div className="relative flex-1">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by case ID, victim name, or district..."
            className="w-full pl-8 pr-3 py-1.5 text-xs border border-surface-border rounded bg-white outline-none focus:border-steel focus:ring-1 focus:ring-steel transition"
          />
          <span className="absolute left-2.5 top-2 text-muted text-xs">🔍</span>
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-2.5 top-1.5 text-xs text-muted hover:text-ink"
            >
              ✕
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          <select
            value={fraudFilter}
            onChange={(e) => setFraudFilter(e.target.value)}
            className="border border-surface-border rounded px-2.5 py-1.5 text-xs bg-white text-ink-600 outline-none focus:border-steel"
          >
            <option value="ALL">All Fraud Types</option>
            {fraudTypes.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>

          <select
            value={priorityFilter}
            onChange={(e) => setPriorityFilter(e.target.value)}
            className="border border-surface-border rounded px-2.5 py-1.5 text-xs bg-white text-ink-600 outline-none focus:border-steel"
          >
            <option value="ALL">All Amounts</option>
            <option value="HIGH">High Risk (&gt; ₹50k)</option>
            <option value="MEDIUM">Medium (₹15k – ₹50k)</option>
            <option value="STANDARD">Standard (&lt; ₹15k)</option>
          </select>
        </div>
      </div>

      {/* Table section */}
      {complaints.length === 0 ? (
        <div className="bg-surface-panel border border-t-0 border-surface-border rounded-b-lg px-6 py-12 text-center">
          <p className="font-display text-base font-semibold text-ink">No complaints logged in this session</p>
          <p className="text-xs text-muted mt-1 max-w-sm mx-auto">
            Submit a cybercrime report using the complaint intake form to forecast ATM cash withdrawal zones.
          </p>
          <button
            onClick={onNew}
            className="mt-4 bg-ink text-white text-xs font-medium px-4 py-2 rounded hover:bg-ink-800 transition-colors"
          >
            File First Complaint
          </button>
        </div>
      ) : filtered.length === 0 ? (
        <div className="bg-surface-panel border border-t-0 border-surface-border rounded-b-lg px-6 py-10 text-center">
          <p className="text-sm font-medium text-ink">No matching complaints found</p>
          <p className="text-xs text-muted mt-1">Try clearing your search query or selecting a different filter.</p>
          <button
            onClick={() => {
              setSearch("");
              setFraudFilter("ALL");
              setPriorityFilter("ALL");
            }}
            className="mt-3 text-xs text-steel font-medium hover:underline"
          >
            Reset all filters
          </button>
        </div>
      ) : (
        <div className="bg-surface-panel border border-t-0 border-surface-border rounded-b-lg overflow-x-auto shadow-xs">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="bg-surface text-muted uppercase tracking-wider font-semibold border-b border-surface-border text-[11px]">
                <th className="px-4 py-2.5">Case ID</th>
                <th className="px-4 py-2.5">Victim</th>
                <th className="px-4 py-2.5">Fraud Type</th>
                <th className="px-4 py-2.5">Amount (₹)</th>
                <th className="px-4 py-2.5">Priority</th>
                <th className="px-4 py-2.5">Forecasted ATM Zone</th>
                <th className="px-4 py-2.5">Filed At</th>
                <th className="px-4 py-2.5 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border">
              {filtered.map((c) => {
                const amt = Number(c.amount) || 0;
                const topZone = c.zones && c.zones[0];
                return (
                  <tr
                    key={c.id}
                    onClick={() => onSelect(c)}
                    className="hover:bg-ink-50/50 cursor-pointer transition-colors"
                  >
                    <td className="px-4 py-3 font-mono font-medium text-steel">{c.id}</td>
                    <td className="px-4 py-3">
                      <div className="font-medium text-ink">{c.victim || "Anonymous"}</div>
                      {c.victim_district && <div className="text-[11px] text-muted">{c.victim_district}</div>}
                    </td>
                    <td className="px-4 py-3 text-ink-600">{c.fraudType}</td>
                    <td className="px-4 py-3 font-mono font-medium text-ink">
                      ₹{amt.toLocaleString("en-IN")}
                    </td>
                    <td className="px-4 py-3">{getPriorityBadge(amt)}</td>
                    <td className="px-4 py-3">
                      {topZone ? (
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-ink">{topZone.name}</span>
                          <span className="font-mono text-[11px] text-risk font-semibold">
                            {Math.round(topZone.probability * 100)}%
                          </span>
                        </div>
                      ) : (
                        <span className="text-muted text-[11px]">Analyzing...</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted font-mono text-[11px]">
                      {formatTime(c.filedAt)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="text-[11px] text-steel font-medium hover:underline">
                        View Map →
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
