export const datasetBExperimentalEnabled = String(
  import.meta.env.VITE_ENABLE_DATASET_B_EXPERIMENTAL || "false",
).toLowerCase() === "true";
