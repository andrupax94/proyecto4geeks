"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from "recharts";
import { ClassDistributionItem } from "@/types";

interface EDAChartProps {
  data: ClassDistributionItem[];
  title: string;
  color?: string;
}

export default function EDAChart({ data, title, color = "#3b82f6" }: EDAChartProps) {
  const sorted = [...data].sort((a, b) => b.count - a.count);

  return (
    <div className="bg-white rounded-xl shadow p-5">
      <h3 className="text-base font-semibold text-gray-700 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 20, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis
            dataKey="class"
            type="category"
            width={140}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(value: number) => [value.toLocaleString(), "Audios"]}
          />
          <Bar dataKey="count" fill={color} radius={[0, 4, 4, 0]}>
            {sorted.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={color}
                fillOpacity={1 - index * 0.04}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
