const items = [
  { key: "form", label: "File complaint" },
  { key: "results", label: "All complaints" },
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
            className={`w-full text-left px-5 py-3 text-sm transition-colors ${
              active === item.key
                ? "bg-ink-800 text-white border-l-2 border-risk"
                : "text-ink-100 hover:bg-ink-800/60"
            }`}
          >
            {item.label}
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
