import { FlaskConical, MapPin, RotateCcw, Sparkles } from "lucide-react";
import { useState } from "react";
import Reveal from "../components/Reveal";
import { predictDatasetBExperimental } from "../services/api";

const initialForm = {
  brand: "",
  model_family: "",
  manufacturing_year: "",
  kms_driven: "",
  fuel_type: "",
  city: "",
};

const fields = [
  ["brand", "Brand", "e.g. Toyota", "text"],
  ["model_family", "Model family", "e.g. Innova", "text"],
  ["manufacturing_year", "Manufacturing year", "1980–2025", "number"],
  ["kms_driven", "Kilometres driven", "e.g. 45000", "number"],
  ["fuel_type", "Fuel type", "e.g. Diesel", "text"],
  ["city", "City", "e.g. Delhi", "text"],
];

export default function DatasetBExperimentalPage() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const update = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  };

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setResult(null);
    const payload = {
      ...form,
      manufacturing_year: Number(form.manufacturing_year),
      kms_driven: Number(form.kms_driven),
    };
    if (!payload.brand || !payload.model_family || !payload.fuel_type || !payload.city
      || !Number.isInteger(payload.manufacturing_year)
      || payload.manufacturing_year < 1980 || payload.manufacturing_year > 2025
      || !Number.isFinite(payload.kms_driven) || payload.kms_driven < 0) {
      setError("Enter all six fields. Use a year from 1980 to 2025 and non-negative kilometres.");
      return;
    }
    setLoading(true);
    try {
      setResult(await predictDatasetBExperimental(payload));
    } catch (requestError) {
      setError(requestError.message || "The experimental model is unavailable.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
  };

  return (
    <div className="page experimental-page">
      <Reveal className="page-heading">
        <div>
          <span className="eyebrow">EXPERIMENTAL / DATASET B</span>
          <h2>Explore the Dataset B model</h2>
          <p>Test an isolated model trained on the approved Kaggle used-car dataset.</p>
        </div>
        <div className="heading-mark"><FlaskConical size={25} /></div>
      </Reveal>
      <div className="experimental-notice">
        <Sparkles size={17} />
        <span><strong>Experimental only.</strong> This estimate is not production-ready and the interval is not a guaranteed selling-price range.</span>
      </div>
      <div className="experimental-grid">
        <Reveal>
          <form className="vehicle-form" onSubmit={submit} noValidate>
            <div className="form-heading">
              <div className="form-heading-icon"><MapPin size={16} /></div>
              <div><strong>Dataset B vehicle profile</strong><span>Only fields available in this dataset</span></div>
            </div>
            <div className="form-grid">
              {fields.map(([name, label, placeholder, type]) => (
                <label key={name}>
                  <span>{label}</span>
                  <div className="control">
                    <input name={name} type={type} value={form[name]} onChange={update} placeholder={placeholder} min={type === "number" ? (name === "kms_driven" ? 0 : 1980) : undefined} max={name === "manufacturing_year" ? 2025 : undefined} />
                  </div>
                </label>
              ))}
            </div>
            {error && <div className="alert error-alert experimental-error" role="alert">{error}</div>}
            <div className="form-actions">
              <button className="button-primary" type="submit" disabled={loading}>{loading ? "Estimating..." : "Estimate with Dataset B"}</button>
              <button className="button-ghost" type="button" onClick={reset}><RotateCcw size={13} /> Reset</button>
            </div>
          </form>
        </Reveal>
        <Reveal className="experimental-result-card">
          {!result ? (
            <div className="result-empty experimental-empty"><div className="empty-sparkle"><Sparkles size={18} /></div><h3>Your experimental estimate will appear here</h3><p>Complete the six-field profile to test the isolated Dataset B model.</p></div>
          ) : (
            <div className="experimental-result">
              <span className="overline">EXPERIMENTAL ESTIMATE</span>
              <strong className="experimental-price">{result.predicted_price_lakhs.toFixed(2)} <small>lakhs</small></strong>
              <div className="experimental-interval"><span>Experimental prediction interval</span><strong>{result.lower_bound_lakhs.toFixed(2)} – {result.upper_bound_lakhs.toFixed(2)} lakhs</strong></div>
              <div className="experimental-meta"><span>Model identifier</span><strong>{result.model_identifier}</strong></div>
              {result.warning && <div className="experimental-warning">{result.warning}</div>}
              <p className="experimental-disclaimer">This interval reflects held-out calibration evidence and is not a guarantee for this individual vehicle.</p>
            </div>
          )}
        </Reveal>
      </div>
    </div>
  );
}
