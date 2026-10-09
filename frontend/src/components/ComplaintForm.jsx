import { useState } from "react";

const fraudTypes = [
  "UPI fraud",
  "Card cloning / skimming",
  "OTP / phishing",
  "Fake loan app",
  "Investment / trading scam",
  "SIM swap",
  "Other",
];

const inputBase =
  "w-full border rounded px-3 py-2 text-sm bg-white outline-none focus:ring-1 focus:ring-steel";
const inputOk = "border-surface-border";
const inputError = "border-risk";

const emptyForm = {
  victim: "",
  contact: "",
  fraudType: "",
  amount: "",
  description: "",
};

export default function ComplaintForm() {
  const [form, setForm] = useState(emptyForm);
  const [errors, setErrors] = useState({});
  const [submitted, setSubmitted] = useState(null);

  function update(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
    if (errors[field]) setErrors((e) => ({ ...e, [field]: undefined }));
  }

  function validate() {
    const next = {};
    if (!form.victim.trim()) next.victim = "Enter the victim's name.";
    if (!/^\d{10}$/.test(form.contact.trim()))
      next.contact = "Enter a 10-digit mobile number.";
    if (!form.fraudType) next.fraudType = "Select a fraud type.";
    if (!form.amount || Number(form.amount) <= 0)
      next.amount = "Enter the amount involved.";
    if (!form.description.trim() || form.description.trim().length < 20)
      next.description = "Add at least 20 characters describing what happened.";
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!validate()) return;
    const id = `CC-2026-${Math.floor(10000 + Math.random() * 89999)}`;
    setSubmitted(id);
    setForm(emptyForm);
  }

  if (submitted) {
    return (
      <div className="max-w-xl">
        <div className="bg-safe-50 border border-safe text-ink rounded px-5 py-4">
          <p className="font-mono text-sm text-safe-500">{submitted}</p>
          <p className="mt-1 font-medium">Complaint filed</p>
          <p className="text-sm text-muted mt-1">
            The prediction pipeline will process this complaint and update risk
            zones within a few minutes.
          </p>
        </div>
        <button
          onClick={() => setSubmitted(null)}
          className="mt-4 text-sm text-steel hover:underline"
        >
          File another complaint
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-xl">
      <h1 className="font-display text-2xl mb-1">File a cybercrime complaint</h1>
      <p className="text-sm text-muted mb-6">
        Details feed the prediction model to forecast likely cash withdrawal zones.
      </p>

      <Field label="Victim name" error={errors.victim}>
        <input
          className={`${inputBase} ${errors.victim ? inputError : inputOk}`}
          value={form.victim}
          onChange={(e) => update("victim", e.target.value)}
          placeholder="Full name as on ID"
        />
      </Field>

      <Field label="Mobile number" error={errors.contact}>
        <input
          className={`${inputBase} ${errors.contact ? inputError : inputOk}`}
          value={form.contact}
          onChange={(e) => update("contact", e.target.value)}
          placeholder="10-digit mobile number"
          inputMode="numeric"
        />
      </Field>

      <Field label="Fraud type" error={errors.fraudType}>
        <select
          className={`${inputBase} ${errors.fraudType ? inputError : inputOk}`}
          value={form.fraudType}
          onChange={(e) => update("fraudType", e.target.value)}
        >
          <option value="">Select type</option>
          {fraudTypes.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Amount involved (₹)" error={errors.amount}>
        <input
          className={`${inputBase} ${errors.amount ? inputError : inputOk}`}
          value={form.amount}
          onChange={(e) => update("amount", e.target.value)}
          placeholder="0"
          inputMode="numeric"
        />
      </Field>

      <Field label="What happened" error={errors.description}>
        <textarea
          className={`${inputBase} ${errors.description ? inputError : inputOk}`}
          value={form.description}
          onChange={(e) => update("description", e.target.value)}
          placeholder="Describe the transaction, how the fraud occurred, and any suspicious links or numbers involved."
          rows={5}
        />
      </Field>

      <button
        type="submit"
        className="mt-2 bg-ink text-white text-sm font-medium px-5 py-2.5 rounded hover:bg-ink-800 transition-colors"
      >
        Submit complaint
      </button>
    </form>
  );
}

function Field({ label, error, children }) {
  return (
    <div className="mb-4">
      <label className="block text-sm text-ink-600 mb-1.5">{label}</label>
      {children}
      {error && <p className="text-xs text-risk mt-1">{error}</p>}
    </div>
  );
}
