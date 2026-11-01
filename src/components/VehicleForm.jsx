import { LoaderCircle, RotateCcw, Sparkles } from "lucide-react";
import { useState } from "react";

const MIN_SUPPORTED_YEAR = 2007;
const MAX_SUPPORTED_YEAR = 2023;
const YEAR_ERROR = "Year must be between the supported dataset years and 2023.";
const parseRegistrationYear = (value) => {
  const text = String(value).trim();
  const shortYear = text.match(/^[A-Za-z]{3}-(\d{2})$/);
  if (shortYear) {
    const year = Number(shortYear[1]);
    return year <= 30 ? 2000 + year : 1900 + year;
  }
  const year = Number(text);
  if (!Number.isInteger(year)) return null;
  return year >= 0 && year <= 30 ? 2000 + year : year;
};

export const initialForm = {
  car_name: "2018 Hyundai Creta 1.6 SX", manufacturing_year: "2018", registration_year: "2018",
  fuel_type: "Petrol", seats: "5", kms_driven: "42000", ownership: "First Owner",
  transmission: "Manual", mileage: "16.8", engine: "1591", insurance_validity: "Comprehensive",
};

const fields = [
  ["car_name", "Car name / model", "text", "e.g. Honda City V CVT"],
  ["manufacturing_year", "Manufacturing year", "number", ""],
  ["registration_year", "Registration year", "text", "e.g. 2018 or Jul-18"],
  ["fuel_type", "Fuel type", "select", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]],
  ["seats", "Number of seats", "number", ""],
  ["kms_driven", "Kilometres driven", "number", ""],
  ["ownership", "Ownership", "select", ["First Owner", "Second Owner", "Third Owner", "Fourth Owner"]],
  ["transmission", "Transmission", "select", ["Manual", "Automatic", "CVT", "AMT"]],
  ["mileage", "Mileage", "number", ""],
  ["engine", "Engine capacity", "number", ""],
  ["insurance_validity", "Insurance validity", "select", ["Comprehensive", "Third Party insurance", "Zero Dep", "Not Available", "Expired"]],
];

function validate(form) {
  const errors = {};
  if (!form.car_name.trim()) errors.car_name = "Enter a car name.";
  const year = Number(form.manufacturing_year);
  if (!Number.isInteger(year) || year < MIN_SUPPORTED_YEAR || year > MAX_SUPPORTED_YEAR) errors.manufacturing_year = YEAR_ERROR;
  if (!form.registration_year.trim()) errors.registration_year = "Enter a registration year.";
  const registrationYear = parseRegistrationYear(form.registration_year);
  if (form.registration_year.trim() && (registrationYear === null || registrationYear < MIN_SUPPORTED_YEAR || registrationYear > MAX_SUPPORTED_YEAR)) {
    errors.registration_year = YEAR_ERROR;
  } else if (registrationYear !== null && Number.isInteger(year) && registrationYear < year) {
    errors.registration_year = "Registration year cannot be earlier than manufacturing year.";
  }
  const seats = Number(form.seats);
  if (!Number.isInteger(seats) || seats < 1 || seats > 20) errors.seats = "Use 1–20 seats.";
  if (!Number.isFinite(Number(form.kms_driven)) || Number(form.kms_driven) < 0) errors.kms_driven = "Enter a valid distance.";
  if (form.mileage && (Number(form.mileage) < 0 || Number(form.mileage) > 200)) errors.mileage = "Use 0–200 km/l.";
  if (form.engine && (Number(form.engine) < 0 || Number(form.engine) > 20000)) errors.engine = "Use 0–20,000 cc.";
  return errors;
}

export default function VehicleForm({ value, onChange, onSubmit, onReset, loading, submitError }) {
  const [errors, setErrors] = useState({});
  const submit = (event) => {
    event.preventDefault();
    const nextErrors = validate(value);
    setErrors(nextErrors);
    if (!Object.keys(nextErrors).length) onSubmit({
      ...value, manufacturing_year: Number(value.manufacturing_year), seats: Number(value.seats),
      kms_driven: Number(value.kms_driven), mileage: value.mileage === "" ? null : Number(value.mileage),
      engine: value.engine === "" ? null : Number(value.engine),
    });
  };
  const change = (key, next) => { onChange({ ...value, [key]: next }); setErrors({ ...errors, [key]: "" }); };
  return <form className="vehicle-form" onSubmit={submit} noValidate><div className="form-heading"><div className="form-heading-icon"><CarIcon /></div><div><strong>Vehicle profile</strong><span>Enter details as they appear on your documents.</span><small className="year-notice">Only cars up to model year 2023 are supported.</small></div></div><div className="form-grid">
    {fields.map(([key, label, type, options]) => <label className={key === "car_name" ? "wide-field" : ""} key={key}><span>{label}</span><div className={`control ${errors[key] ? "has-error" : ""}`}>{type === "select" ? <select value={value[key] || ""} onChange={(event) => change(key, event.target.value)}><option value="">Not specified</option>{options.map((option) => <option key={option}>{option}</option>)}</select> : <input value={value[key] || ""} type={type} placeholder={options} onChange={(event) => change(key, event.target.value)} min={key === "seats" ? 1 : key === "manufacturing_year" ? MIN_SUPPORTED_YEAR : undefined} max={key === "manufacturing_year" || key === "registration_year" ? MAX_SUPPORTED_YEAR : undefined} />}{["kms_driven", "mileage", "engine"].includes(key) && <em>{key === "kms_driven" ? "km" : key === "mileage" ? "km/l" : "cc"}</em>}</div>{errors[key] && <small className="input-error-text">{errors[key]}</small>}</label>)}
  </div>{submitError && <div className="alert error-alert">{submitError}</div>}<div className="form-actions"><button className="button-primary" disabled={loading}>{loading ? <><LoaderCircle className="spin" size={17} /> Predicting...</> : <><Sparkles size={17} /> Predict resale price</>}</button><button type="button" className="button-ghost" onClick={() => { onReset(); setErrors({}); }}><RotateCcw size={15} /> Reset</button></div></form>;
}

function CarIcon() { return <span className="mini-car">▰</span>; }
