import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
  headers: { "Content-Type": "application/json" },
  timeout: 15000,
});

const getErrorMessage = (error) => {
  const detail = error.response?.data?.detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join(". ");
  return detail || error.message || "Unable to connect to the prediction service.";
};

const request = async (promise) => {
  try {
    return (await promise).data;
  } catch (error) {
    throw new Error(getErrorMessage(error));
  }
};

export const getHealth = () => request(api.get("/api/health"));
export const getModelInfo = () => request(api.get("/api/model-info"));
export const getMetrics = () => request(api.get("/api/metrics"));
export const predictPrice = (payload) => request(api.post("/api/predict", payload));
export const predictDatasetBExperimental = (payload) => request(
  api.post("/api/experimental/dataset-b/predict", payload),
);
export default api;
