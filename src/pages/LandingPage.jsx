import { useEffect } from "react";
import { ArrowRight, BarChart3, CarFront, Check, ChevronDown, CircleGauge, Sparkles } from "lucide-react";
import Reveal from "../components/Reveal";

const steps = [
  ["01", "Enter your vehicle details.", "Share the essentials: model, year, usage and condition signals."],
  ["02", "AI analyses the profile.", "A trained XGBoost model weighs your vehicle against market patterns."],
  ["03", "Receive a smarter starting point.", "Get a transparent resale estimate in seconds."],
];

export default function LandingPage({ onNavigate, focusHow = false }) {
  useEffect(() => {
    if (focusHow) window.setTimeout(() => document.getElementById("how-it-works")?.scrollIntoView(), 0);
  }, [focusHow]);
  return (
    <div className="landing-page">
      <section className="landing-hero">
        <Reveal className="hero-copy">
          <span className="landing-eyebrow"><span /> INTELLIGENT CAR VALUATION</span>
          <h2>Know what<br /><em>your car</em> is worth.</h2>
          <p>Discover your vehicle&apos;s estimated resale value with intelligent machine learning. Get a smarter starting point for your next car sale.</p>
          <div className="hero-actions">
            <button className="black-button" onClick={() => onNavigate("predict")}>Get Your Valuation <ArrowRight size={16} /></button>
            <button className="text-button" onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}>Explore How It Works <ChevronDown size={16} /></button>
          </div>
          <div className="hero-proof"><span><Check size={14} /> No sign-up</span><span><Check size={14} /> Real model estimate</span><span><Check size={14} /> Free to use</span></div>
        </Reveal>
        <Reveal className="hero-visual" delay={120}>
          <div className="hero-arch" />
          <img
            src="https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1400&q=85"
            alt="Premium sports car in soft studio light"
            onError={(event) => { event.currentTarget.style.display = "none"; }}
          />
          <div className="hero-visual-caption"><span className="caption-line" /><span>Built for better<br />car decisions.</span></div>
          <div className="hero-orbit"><CircleGauge size={17} /><span>Data-backed<br /><strong>valuation</strong></span></div>
        </Reveal>
      </section>

      <section className="how-section" id="how-it-works">
        <Reveal className="section-lead"><span className="landing-eyebrow">HOW IT WORKS</span><h3>Simple inputs.<br /><em>Smarter direction.</em></h3></Reveal>
        <div className="steps-grid">{steps.map(([number, title, copy], index) => <Reveal className="step-card" delay={index * 80} key={number}><span className="step-number">{number}</span><div className="step-icon">{number === "01" ? <CarFront size={19} /> : number === "02" ? <Sparkles size={19} /> : <BarChart3 size={19} />}</div><h4>{title}</h4><p>{copy}</p></Reveal>)}</div>
      </section>

      <section className="why-section">
        <Reveal><span className="landing-eyebrow">WHY AUTOVALUE AI</span><h3>A clearer way to<br /><em>move forward.</em></h3></Reveal>
        <Reveal className="why-points"><div><strong>01</strong><span>Machine-learning-powered valuation that uses your vehicle profile, not a generic guess.</span></div><div><strong>02</strong><span>A simple, spacious form that keeps the focus on the information that matters.</span></div><div><strong>03</strong><span>Instant results from a deployed XGBoost model, with the evidence to understand it.</span></div></Reveal>
      </section>

      <Reveal className="landing-cta"><span className="landing-eyebrow">YOUR NEXT MOVE</span><h3>Ready to discover<br />your car&apos;s value?</h3><button className="black-button" onClick={() => onNavigate("predict")}>Calculate My Car&apos;s Value <ArrowRight size={16} /></button></Reveal>
    </div>
  );
}
