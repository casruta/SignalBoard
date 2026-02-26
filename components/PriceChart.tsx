"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
  Bar,
} from "recharts";
import type { PricePoint } from "@/lib/types";

export function PriceChart({ data }: { data: PricePoint[] }) {
  const minPrice = Math.min(...data.map((d) => Math.min(d.close, d.predicted))) * 0.995;
  const maxPrice = Math.max(...data.map((d) => Math.max(d.close, d.predicted))) * 1.005;

  return (
    <div className="signal-card p-6">
      <h3 className="text-lg font-semibold mb-4">
        Price vs. ML Predicted — 90 Day Window
      </h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#8892a4", fontSize: 11 }}
              tickFormatter={(v) => v.slice(5)}
              interval={9}
            />
            <YAxis
              yAxisId="price"
              domain={[minPrice, maxPrice]}
              tick={{ fill: "#8892a4", fontSize: 11 }}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            <YAxis
              yAxisId="volume"
              orientation="right"
              tick={{ fill: "#8892a4", fontSize: 11 }}
              tickFormatter={(v) => `${(v / 1e6).toFixed(0)}M`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111827",
                border: "1px solid #1e2d3d",
                borderRadius: "0.5rem",
                color: "#e2e8f0",
              }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={((value: any, name: any) => {
                const v = Number(value) || 0;
                const n = String(name ?? "");
                if (n === "volume")
                  return [`${(v / 1e6).toFixed(1)}M`, "Volume"];
                return [`$${v.toFixed(2)}`, n === "close" ? "Actual" : "ML Predicted"];
              }) as never}
            />
            <Legend
              wrapperStyle={{ color: "#8892a4", fontSize: 12 }}
            />
            <Bar
              yAxisId="volume"
              dataKey="volume"
              fill="rgba(59, 130, 246, 0.15)"
              name="volume"
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="close"
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="predicted"
              stroke="#22c55e"
              strokeWidth={2}
              strokeDasharray="6 3"
              dot={false}
              name="predicted"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
