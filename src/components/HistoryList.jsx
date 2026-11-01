import { ArrowUpRight, History as HistoryIcon } from "lucide-react";

export default function HistoryList({ history, onView }) {
  if (!history.length) return <div className="empty-history"><HistoryIcon size={22} /><p>No predictions yet</p><small>Successful estimates from this browser session will appear here.</small></div>;
  return <div className="history-list">{history.map((item, index) => <button className="history-row" key={`${item.timestamp}-${index}`} onClick={() => onView(item)}><span className="history-avatar">{(item.vehicle.car_name || "C").slice(0, 1).toUpperCase()}</span><span className="history-main"><strong>{item.vehicle.car_name || "Unnamed vehicle"}</strong><small>{item.vehicle.manufacturing_year} · {item.timestamp}</small></span><strong className="history-value">₹{item.predicted_price_lakhs.toFixed(2)}L</strong><ArrowUpRight size={15} /></button>)}</div>;
}
