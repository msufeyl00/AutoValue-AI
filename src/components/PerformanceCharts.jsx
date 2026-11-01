import { BarChart, Bar, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function PerformanceCharts({ metrics }) {
  const data = [
    { name: "MAE", value: Number(metrics?.mae || 0), suffix: " lakhs" },
    { name: "RMSE", value: Number(metrics?.rmse || 0), suffix: " lakhs" },
    { name: "R²", value: Number(metrics?.r2 || 0), suffix: "" },
    { name: "Within 20%", value: Number(metrics?.within_20_percent || 0) * 100, suffix: "%" },
  ];
  return <div className="chart-card"><div className="chart-heading"><div><p className="overline">Held-out test set</p><h3>Model performance snapshot</h3></div><span>n = {metrics?.test_rows || "—"} rows</span></div><ResponsiveContainer width="100%" height={260}><BarChart data={data} margin={{ top: 10, right: 12, left: -20, bottom: 4 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf4" /><XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: "#8290a4", fontSize: 11 }} /><YAxis axisLine={false} tickLine={false} tick={{ fill: "#8290a4", fontSize: 10 }} /><Tooltip cursor={{ fill: "#f4f7fb" }} formatter={(value, _, item) => [`${Number(value).toFixed(2)}${item.payload.suffix}`, item.name]} /><Bar dataKey="value" radius={[6, 6, 0, 0]}>{data.map((item, index) => <Cell key={item.name} fill={index === 2 ? "#4b8ff5" : "#182d50"} />)}</Bar></BarChart></ResponsiveContainer></div>;
}
