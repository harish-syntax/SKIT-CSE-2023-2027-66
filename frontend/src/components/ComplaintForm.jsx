import { useState } from "react";
import { fraudTypes, victimDistricts } from "../data/zones";

const inputBase =
  "w-full border rounded px-3 py-2 text-sm bg-white outline-none focus:ring-1 focus:ring-steel transition";
const inputOk = "border-surface-border";
const inputError = "border-risk";

function getCurrentTime() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function generateIncidentId() {
  return `INC-${Math.floor(100000 + Math.random() * 900000)}`;
}

export default function ComplaintForm({ onSubmit, submitting }) {
  const [form, setForm] = useState({
    incident_id: generateIncidentId(),
    victim: "",
    contact: "",
    victim_district: "Jaipur",
    fraud_type: "",
    amount_inr: "",
    time_of_complaint: getCurrentTime(),
    description: "",
  });

  const [errors, setErrors] = useState({});

  function update(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
    if (errors[field]) setErrors((e) => ({ ...e, [field]: undefined }));
  }

  function validate() {
    const next = {};
    if (!form.victim.trim()) next.victim = "Enter the victim's name.";
    if (!/^\d{10}$/.test(form.contact.trim()))
      next.contact = "Enter a valid 10-digit mobile number.";
    if (!form.victim_district) next.victim_district = "Select a victim district.";
    if (!form.fraud_type) next.fraud_type = "Select a cybercrime fraud type.";

    // Strict Amount Validation: Numbers only and strictly greater than ₹1,000
    const amtStr = String(form.amount_inr ?? "").trim();
    if (!amtStr) {
      next.amount_inr = "Enter the fraud loss amount in ₹.";
    } else if (!/^\d+$/.test(amtStr)) {
      next.amount_inr = "Amount must contain digits only (no letters or symbols).";
    } else {
      const amtVal = Number(amtStr);
      if (amtVal <= 1000) {
        next.amount_inr = "Amount must be greater than ₹1,000 (minimum fraud reporting threshold).";
      } else if (amtVal > 100000000) {
        next.amount_inr = "Amount exceeds the maximum limit of ₹10,00,00,000.";
      }
    }

    if (!form.description.trim() || form.description.trim().length < 15)
      next.description = "Add at least 15 characters describing what happened.";
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!validate()) return;

    onSubmit({
      ...form,
      amount_inr: Number(form.amount_inr),
      // for backward compatibility with screens expecting amount & fraudType:
      amount: Number(form.amount_inr),
      fraudType: form.fraud_type,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-2xl bg-white p-8 rounded-lg shadow-sm border border-surface-border">
      <div className="flex items-center justify-between mb-1">
        <h1 className="font-display text-2xl text-ink font-semibold">File a Cybercrime Complaint</h1>
        <span className="font-mono text-xs bg-ink-50 text-steel font-medium px-2.5 py-1 rounded">
          {form.incident_id}
        </span>
      </div>
      <p className="text-sm text-muted mb-6">
        Details are dispatched to the FastAPI backend and evaluated by the predictive model to forecast likely ATM withdrawal zones.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Field label="Victim Name" error={errors.victim}>
          <input
            className={`${inputBase} ${errors.victim ? inputError : inputOk}`}
            value={form.victim}
            onChange={(e) => update("victim", e.target.value)}
            placeholder="Full name as on ID"
          />
        </Field>

        <Field label="Mobile Number" error={errors.contact} hint="10 digits only">
          <input
            className={`${inputBase} ${errors.contact ? inputError : inputOk}`}
            value={form.contact}
            onChange={(e) => update("contact", e.target.value.replace(/\D/g, "").slice(0, 10))}
            placeholder="10-digit mobile number"
            inputMode="numeric"
            maxLength={10}
          />
        </Field>

        <Field label="Victim District (Rajasthan)" error={errors.victim_district}>
          <select
            className={`${inputBase} ${errors.victim_district ? inputError : inputOk}`}
            value={form.victim_district}
            onChange={(e) => update("victim_district", e.target.value)}
          >
            {victimDistricts.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </Field>

        <Field label="Fraud Type" error={errors.fraud_type}>
          <select
            className={`${inputBase} ${errors.fraud_type ? inputError : inputOk}`}
            value={form.fraud_type}
            onChange={(e) => update("fraud_type", e.target.value)}
          >
            <option value="">Select fraud type</option>
            {fraudTypes.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </Field>

        <Field
          label="Amount Involved (₹)"
          error={errors.amount_inr}
          hint="Numbers only · Must be > ₹1,000"
        >
          <div className="relative">
            <span className="absolute left-3 top-2 text-ink-400 font-medium text-sm select-none">₹</span>
            <input
              type="text"
              className={`${inputBase} pl-7 ${errors.amount_inr ? inputError : inputOk}`}
              value={form.amount_inr}
              onChange={(e) => {
                // Strict restriction: Allow only digits 0-9 (strips letters, symbols, decimals, negatives)
                const digitsOnly = e.target.value.replace(/\D/g, "");
                update("amount_inr", digitsOnly);
              }}
              placeholder="e.g. 25000 (Min. ₹1,001)"
              inputMode="numeric"
            />
          </div>
        </Field>

        <Field label="Time of Complaint (HH:MM)" error={errors.time_of_complaint}>
          <input
            className={`${inputBase} ${inputOk}`}
            value={form.time_of_complaint}
            onChange={(e) => update("time_of_complaint", e.target.value)}
            placeholder="14:30"
          />
        </Field>
      </div>

      <div className="mt-4">
        <Field label="Incident Details / Modus Operandi" error={errors.description}>
          <textarea
            className={`${inputBase} ${errors.description ? inputError : inputOk}`}
            value={form.description}
            onChange={(e) => update("description", e.target.value)}
            placeholder="Describe transaction details, SMS alerts, UPI handles, or fraudulent links involved..."
            rows={4}
          />
        </Field>
      </div>

      <div className="mt-6 flex items-center justify-between pt-4 border-t border-surface-border">
        <span className="text-xs text-muted flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-safe inline-block"></span>
          Backend API Integration Ready
        </span>

        <button
          type="submit"
          disabled={submitting}
          className="bg-ink text-white text-sm font-medium px-6 py-2.5 rounded hover:bg-ink-800 transition-colors disabled:opacity-60 flex items-center gap-2"
        >
          {submitting ? (
            <>
              <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
              </svg>
              Predicting Zones…
            </>
          ) : (
            "Submit Complaint & Predict ATM Zones"
          )}
        </button>
      </div>
    </form>
  );
}

function Field({ label, error, hint, children }) {
  return (
    <div className="mb-2">
      <div className="flex items-center justify-between mb-1.5">
        <label className="block text-xs font-medium text-ink-600 uppercase tracking-wider">{label}</label>
        {hint && <span className="text-[11px] text-muted font-normal">{hint}</span>}
      </div>
      {children}
      {error && <p className="text-xs text-risk mt-1 font-medium">{error}</p>}
    </div>
  );
}
