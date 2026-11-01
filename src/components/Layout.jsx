import { CarFront, History, Menu, X } from "lucide-react";
import { useEffect, useState } from "react";
import Reveal from "./Reveal";
import { datasetBExperimentalEnabled } from "../config/featureFlags";

const navItems = [
  { id: "dashboard", label: "Home" },
  { id: "how", label: "How it works" },
  { id: "about", label: "About" },
  { id: "history", label: "History" },
];
const experimentalNavItem = { id: "dataset-b-experimental", label: "Dataset B experimental" };

export default function Layout({ activePage, onNavigate, children, predictionCount, modelName, health }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  const navigate = (page) => { onNavigate(page); setMobileOpen(false); };
  const visibleNavItems = datasetBExperimentalEnabled ? [...navItems, experimentalNavItem] : navItems;
  return (
    <div className="app-shell">
      {mobileOpen && <button className="mobile-backdrop" onClick={() => setMobileOpen(false)} aria-label="Close menu" />}
      <section className="main-shell">
        <header className={`topbar ${scrolled ? "topbar-scrolled" : ""}`}><button className="mobile-menu" onClick={() => setMobileOpen((value) => !value)} aria-label="Toggle navigation" aria-expanded={mobileOpen}>{mobileOpen ? <X size={20} /> : <Menu size={20} />}</button><button className="public-brand" onClick={() => navigate("dashboard")}><span className="public-mark"><CarFront size={16} /></span><span>AutoValue <b>AI</b></span></button><nav className={`public-nav ${mobileOpen ? "public-nav-open" : ""}`}>{visibleNavItems.map(({ id, label }) => <button key={id} className={activePage === id ? "public-active" : ""} onClick={() => navigate(id)}>{label}</button>)}<button className="nav-cta" onClick={() => navigate("predict")}>Get started <span>↗</span></button></nav><div className="topbar-session"><History size={14} /> {predictionCount}</div></header>
        <main className="content">{children}</main>
        <Reveal as="footer" className="site-footer"><button onClick={() => navigate("dashboard")}>AutoValue <b>AI</b></button><span>Intelligent valuation for better car decisions.</span><span>© 2026 AutoValue AI</span></Reveal>
      </section>
    </div>
  );
}
