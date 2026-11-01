import { Trash2 } from "lucide-react";
import HistoryList from "../components/HistoryList";
import Reveal from "../components/Reveal";

export default function HistoryPage({ history, onView, onClear }) {
  return <div className="page"><Reveal className="page-heading"><div><span className="eyebrow">SESSION DATA</span><h2>Prediction history</h2><p>Your successful estimates are kept only in the current browser session.</p></div>{history.length > 0 && <button className="button-ghost danger-button" onClick={onClear}><Trash2 size={15} /> Clear history</button>}</Reveal><Reveal className="surface-card"><HistoryList history={history} onView={onView} /></Reveal></div>;
}
