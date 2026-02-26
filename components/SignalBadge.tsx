import type { SignalDirection } from "@/lib/types";

export function SignalBadge({
  direction,
  size = "md",
}: {
  direction: SignalDirection;
  size?: "sm" | "md" | "lg";
}) {
  const sizeClasses = {
    sm: "px-2 py-0.5 text-xs",
    md: "px-3 py-1 text-sm",
    lg: "px-4 py-1.5 text-base",
  };

  return (
    <span
      className={`inline-flex items-center rounded-full font-medium ${sizeClasses[size]} badge-${direction}`}
    >
      {direction === "bullish" && "BULLISH"}
      {direction === "bearish" && "BEARISH"}
      {direction === "neutral" && "NEUTRAL"}
    </span>
  );
}
