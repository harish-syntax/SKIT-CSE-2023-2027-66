function formatTime(iso) {
  return new Date(iso).toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function ResultScreen({ complaints, onSelect, onNew }) {
  const totalAmount = complaints.reduce((s, c) => s + c.amount, 0);

  return (
    <div className="max-w-4xl">
      <h1 className="font-display text-2xl mb-1">All submitted complaints</h1>
      <p className="text-sm text-muted mb-6">
        Every complaint filed in this session, with its top predicted zone.
      </p>

      <div className="grid grid-cols-2 gap-3 max-w-md mb-6">
        <div className="bg-surface-panel border border-surface-border rounded px-4 py-3">
          <p className="text-xs text-muted">Complaints filed</p>
          <p className="font-mono text-xl mt-1">{complaints.length}</p>
        </div>
        <div className="bg-surface-panel border border-surface-border rounded px-4 py-3">
          <p className="text-xs text-muted">Total amount</p>
          <p className="font-mono text-xl mt-1">₹{totalAmount.toLocaleString("en-IN")}</p>
        </div>
      </div>

      {complaints.length === 0 ? (
        <div className="bg-surface-panel border border-surface-border rounded px-6 py-10 text-center">
          <p className="font-medium">No complaints yet</p>
          <p className="text-sm text-muted mt-1">
            File a complaint and its predicted zones will show up here.
          </p>
          <button
            onClick={onNew}
            className="mt-4 bg-ink text-white text-sm font-medium px-5 py-2.5 rounded hover:bg-ink-800 transition-colors"
          >
            File complaint
          </button>
        </div>
      ) : (
        <div className="bg-surface-panel border border-surface-border rounded">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted border-b border-surface-border">
                <th className="px-4 py-2 font-medium">Case ID</th>
                <th className="px-4 py-2 font-medium">Victim</th>
                <th className="px-4 py-2 font-medium">Fraud type</th>
                <th className="px-4 py-2 font-medium">Amount</th>
                <th className="px-4 py-2 font-medium">Top zone</th>
                <th className="px-4 py-2 font-medium">Filed</th>
              </tr>
            </thead>
            <tbody>
              {complaints.map((c) => (
                <tr
                  key={c.id}
                  onClick={() => onSelect(c)}
                  className="border-b border-surface-border last:border-0 cursor-pointer hover:bg-surface"
                >
                  <td className="px-4 py-2.5 font-mono text-xs">{c.id}</td>
                  <td className="px-4 py-2.5">{c.victim}</td>
                  <td className="px-4 py-2.5">{c.fraudType}</td>
                  <td className="px-4 py-2.5 font-mono">₹{c.amount.toLocaleString("en-IN")}</td>
                  <td className="px-4 py-2.5">
                    {c.zones && c.zones[0] ? (
                      <>
                        {c.zones[0].name}{" "}
                        <span className="font-mono text-xs text-risk font-semibold">
                          {Math.round(c.zones[0].probability * 100)}%
                        </span>
                      </>
                    ) : (
                      <span className="text-muted text-xs">Pending</span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-muted">
                    {formatTime(c.filedAt)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
