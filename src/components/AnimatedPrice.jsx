import { useEffect, useState } from "react";

export default function AnimatedPrice({ value }) {
  const [displayValue, setDisplayValue] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setDisplayValue(value);
      return undefined;
    }
    let frame;
    const start = performance.now();
    const duration = 650;
    const tick = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - ((1 - progress) ** 3);
      setDisplayValue(value * eased);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value]);
  return <>{displayValue.toFixed(2)}</>;
}
