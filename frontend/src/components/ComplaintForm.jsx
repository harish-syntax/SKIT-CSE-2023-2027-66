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
    if (!form.amount_inr || Number(form.amount_inr) <= 0)
      next.amount_inr = "Enter a valid loss amount in ₹.";
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

        <Field label="Mobile Number" error={errors.contact}>
          <input
            className={`${inputBase} ${errors.contact ? inputError : inputOk}`}
            value={form.contact}
            onChange={(e) => update("contact", e.target.value)}
            placeholder="10-digit mobile number"
            inputMode="numeric"
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

        <Field label="Amount Involved (₹)" error={errors.amount_inr}>
          <input
            className={`${inputBase} ${errors.amount_inr ? inputError : inputOk}`}
            value={form.amount_inr}
            onChange={(e) => update("amount_inr", e.target.value)}
            placeholder="e.g. 25000"
            inputMode="numeric"
          />
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

function Field({ label, error, children }) {
  return (
    <div className="mb-2">
      <label className="block text-xs font-medium text-ink-600 mb-1.5 uppercase tracking-wider">{label}</label>
      {children}
      {error && <p className="text-xs text-risk mt-1">{error}</p>}
    </div>
  );
}
