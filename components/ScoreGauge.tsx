"use client";

export function ScoreGauge({
  score,
  confidence,
  label,
}: {
  score: number;
  confidence: number;
  label: string;
}) {
  const normalized = (score + 1) / 2; // -1..1 -> 0..1
  const rotation = normalized * 180 - 90; // -90 to 90 degrees
  const color =
    score > 0.3
      ? "var(--color-bullish)"
      : score < -0.3
        ? "var(--color-bearish)"
        : "var(--color-neutral)";

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-32 h-16 overflow-hidden">
        <svg viewBox="0 0 120 60" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 10 55 A 50 50 0 0 1 110 55"
            fill="none"
            stroke="var(--color-border)"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Colored arc */}
          <path
            d="M 10 55 A 50 50 0 0 1 110 55"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${normalized * 157} 157`}
            opacity="0.8"
          />
          {/* Needle */}
          <line
            x1="60"
            y1="55"
            x2={60 + 35 * Math.cos((rotation * Math.PI) / 180)}
            y2={55 - 35 * Math.sin((-rotation * Math.PI) / 180)}
            stroke={color}
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle cx="60" cy="55" r="3" fill={color} />
        </svg>
      </div>
      <div className="text-center">
        <div className="text-lg font-bold" style={{ color }}>
          {score > 0 ? "+" : ""}
          {score.toFixed(2)}
        </div>
        <div className="text-xs text-[var(--color-text-dim)]">{label}</div>
        <div className="text-xs text-[var(--color-text-dim)]">
          {(confidence * 100).toFixed(0)}% confidence
        </div>
      </div>
    </div>
  );
}
