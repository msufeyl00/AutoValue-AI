export default function StatCard({ label, value, detail, icon: Icon, tone = "" }) {
  return <article className={`stat-card ${tone}`}><div className="stat-card-top"><span>{label}</span>{Icon && <Icon size={17} />}</div><strong>{value}</strong><small>{detail}</small></article>;
}
