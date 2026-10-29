const items = [
  { key: "form", label: "File complaint", icon: "📝" },
  { key: "results", label: "All complaints", icon: "📋" },
  { key: "alerts", label: "Patrol Alerts", icon: "🚨", badge: "LIVE" },
];

export default function Sidebar({ active, onNavigate }) {
  return (
    <aside className="w-56 shrink-0 bg-ink text-ink-50 flex flex-col">
      <div className="px-5 py-6 border-b border-ink-600">
        <p className="font-display text-lg leading-tight">ATM Fraud<br />Tracker</p>
        <p className="text-xs text-ink-400 mt-1 font-mono">console v0.1</p>
      </div>
      <nav className="flex-1 py-3">
        {items.map((item) => (
          <button
            key={item.key}
            onClick={() => onNavigate(item.key)}
            className={`w-full text-left px-5 py-3 text-sm transition-colors flex items-center justify-between ${
              active === item.key
                ? "bg-ink-800 text-white border-l-2 border-risk font-medium"
                : "text-ink-100 hover:bg-ink-800/60"
            }`}
          >
            <span className="flex items-center gap-2">
              <span className="text-xs">{item.icon}</span>
              {item.label}
            </span>
            {item.badge && (
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-risk text-white uppercase tracking-wider animate-pulse">
                {item.badge}
              </span>
            )}
          </button>
        ))}
      </nav>
      <div className="px-5 py-4 border-t border-ink-600 text-xs text-ink-400">
        <p>Signed in as</p>
        <p className="text-ink-100 mt-0.5">Insp. H. Sharma</p>
      </div>
    </aside>
  );
}
