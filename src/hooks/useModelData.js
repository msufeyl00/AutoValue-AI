import { useCallback, useEffect, useState } from "react";
import { getHealth, getMetrics, getModelInfo } from "../services/api";

export function useModelData() {
  const [data, setData] = useState({ health: null, metrics: null, modelInfo: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [health, metrics, modelInfo] = await Promise.all([
        getHealth(), getMetrics(), getModelInfo(),
      ]);
      setData({ health, metrics, modelInfo });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  return { ...data, loading, error, refresh };
}
