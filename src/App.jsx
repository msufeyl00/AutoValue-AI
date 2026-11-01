import { lazy, Suspense, useMemo, useState } from "react";
import Layout from "./components/Layout";
import LandingPage from "./pages/LandingPage";
import { useModelData } from "./hooks/useModelData";
import { datasetBExperimentalEnabled } from "./config/featureFlags";

const PredictPrice = lazy(() => import("./pages/PredictPrice"));
const HistoryPage = lazy(() => import("./pages/HistoryPage"));
const PerformancePage = lazy(() => import("./pages/PerformancePage"));
const AboutPage = lazy(() => import("./pages/AboutPage"));
const DatasetBExperimentalPage = lazy(() => import("./pages/DatasetBExperimentalPage"));

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [history, setHistory] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const { health, metrics, modelInfo, loading, error } = useModelData();
  const savePrediction = (prediction) => setHistory((current) => [{ ...prediction, timestamp: new Date().toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }) }, ...current]);
  const page = useMemo(() => {
    const common = { metrics, modelInfo };
    if (activePage === "predict") return <PredictPrice onSave={savePrediction} initialResult={selectedPrediction} />;
    if (activePage === "how") return <LandingPage onNavigate={setActivePage} focusHow />;
    if (activePage === "history") return <HistoryPage history={history} onView={(item) => { setSelectedPrediction(item); setActivePage("predict"); }} onClear={() => setHistory([])} />;
    if (activePage === "performance") return <PerformancePage {...common} />;
    if (activePage === "about") return <AboutPage modelInfo={modelInfo} onNavigate={setActivePage} />;
    if (activePage === "dataset-b-experimental" && datasetBExperimentalEnabled) {
      return <DatasetBExperimentalPage />;
    }
    return <LandingPage onNavigate={setActivePage} />;
  }, [activePage, history, health, metrics, modelInfo]);
  return <Layout activePage={activePage} onNavigate={setActivePage} predictionCount={history.length} modelName={modelInfo?.algorithm} health={health}>{loading && !modelInfo ? <div className="loading-screen"><span className="spinner" /><p>Connecting to the model service...</p></div> : error && !modelInfo ? <div className="page"><div className="alert error-alert">{error}</div><button className="button-primary" onClick={() => window.location.reload()}>Retry connection</button></div> : <Suspense fallback={<div className="loading-screen"><span className="spinner" /><p>Loading this section...</p></div>}>{page}</Suspense>}</Layout>;
}
