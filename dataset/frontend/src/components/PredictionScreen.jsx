import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";

function riskColor(p) {
  if (p >= 0.8) return "#C1432D";
  if (p >= 0.6) return "#1D4E89";
  return "#5B6B7A";
}

export default function PredictionScreen({ complaint, zones, onViewAll, onNew }) {
  const center = [zones[0].lat, zones[0].lng];

  return (
    <div className="max-w-4xl">
      <p className="font-mono text-xs text-muted">{complaint.id}</p>
      <h1 className="font-display text-2xl mb-1">Top 3 predicted ATM zones</h1>
      <p className="text-sm text-muted mb-6">
        Likely cash withdrawal locations for this {complaint.fraudType.toLowerCase()} complaint
        (₹{complaint.amount.toLocaleString("en-IN")}), ranked by model probability.
      </p>

      <div className="grid grid-cols-5 gap-4">
        <div className="col-span-3 h-[380px] bg-surface-panel border border-surface-border rounded overflow-hidden">
          <MapContainer center={center} zoom={12} style={{ height: "100%", width: "100%" }}>
            <TileLayer
              attribution="&copy; OpenStreetMap contributors"
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {zones.map((z, i) => (
              <CircleMarker
                key={z.id}
                center={[z.lat, z.lng]}
                radius={10 + z.probability * 14}
                pathOptions={{
                  color: riskColor(z.probability),
                  fillColor: riskColor(z.probability),
                  fillOpacity: 0.35,
                }}
              >
                <Tooltip permanent direction="center">{i + 1}</Tooltip>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>

        <div className="col-span-2 flex flex-col gap-3">
          {zones.map((z, i) => (
            <div
              key={z.id}
              className="bg-surface-panel border border-surface-border rounded px-4 py-3"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-muted">#{i + 1} · {z.id}</span>
                <span
                  className="font-mono text-sm font-medium"
                  style={{ color: riskColor(z.probability) }}
                >
                  {Math.round(z.probability * 100)}%
                </span>
              </div>
              <p className="text-sm font-medium mt-1">{z.name}</p>
              <div className="h-1.5 bg-surface-border rounded mt-2">
                <div
                  className="h-1.5 rounded"
                  style={{
                    width: `${z.probability * 100}%`,
                    background: riskColor(z.probability),
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-3 mt-6">
        <button
          onClick={onViewAll}
          className="bg-ink text-white text-sm font-medium px-5 py-2.5 rounded hover:bg-ink-800 transition-colors"
        >
          View all complaints
        </button>
        <button
          onClick={onNew}
          className="border border-surface-border bg-white text-sm px-5 py-2.5 rounded hover:border-ink-400 transition-colors"
        >
          File another complaint
        </button>
      </div>
    </div>
  );
}
