import { useEffect, useState } from "react";
import { CarFront } from "lucide-react";
import VehicleForm, { initialForm } from "../components/VehicleForm";
import PredictionCard from "../components/PredictionCard";
import Reveal from "../components/Reveal";
import { predictPrice } from "../services/api";

export default function PredictPrice({ onSave, initialResult }) {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(initialResult || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (initialResult) {
      setResult(initialResult);
      setForm((current) => ({ ...current, ...initialResult.vehicle }));
    }
  }, [initialResult]);
  const submit = async (payload) => { setLoading(true); setError(""); try { const response = await predictPrice(payload); setResult(response); onSave(response); } catch (requestError) { setError(requestError.message); } finally { setLoading(false); } };
  return <div className="page"><Reveal className="page-heading"><div><span className="eyebrow">PREDICT PRICE / VEHICLE PROFILE</span><h2>What is your car worth?</h2><p>Provide the details below for a live estimate from the deployed XGBoost model.</p><p className="year-notice">Currently supporting cars manufactured and registered from 2007 through 2023.</p></div><div className="heading-mark"><CarFront size={25} /></div></Reveal><div className="predict-grid"><Reveal><VehicleForm value={form} onChange={setForm} onSubmit={submit} onReset={() => { setForm(initialForm); setResult(null); setError(""); }} loading={loading} submitError={error} /></Reveal><aside className="prediction-panel"><div className="panel-title"><span><span className="live-dot" /> Prediction result</span></div><PredictionCard result={result} form={form} /></aside></div></div>;
}
