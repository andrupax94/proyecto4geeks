"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

import colors, { DARK_COLORS } from "@/services/colors";

interface DistributionChartProps {
  data: { name: string; value: number }[];
  title: string;
  dataKeyName: string;
  dataKeyValue: string;
  color?: string;
}

export default function DistributionChart({
  data = [],
  title,
  dataKeyName,
  dataKeyValue,
  color = "#3b82f6",
}: DistributionChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="box black-box p-5 h-[350px] flex flex-col items-center justify-center transition-all duration-200 hover:blue-box">
        <h3 className="text-base font-semibold text-gray-200 mb-4 self-start">
          {title}
        </h3>
        <p className="text-gray-400 text-sm">
          No hay datos disponibles para este análisis.
        </p>
      </div>
    );
  }

  const sortedData = [...data].sort((a, b) => b.value - a.value);

  return (
    <div className="box black-box p-5 shadow transition-all duration-200 hover:blue-box">
      <h3 className="text-base font-semibold text-gray-200 mb-4">{title}</h3>

      <div className="box blue-box p-3 h-[300px] w-full transition-all duration-200 hover:shadow-lg">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sortedData}
            layout="vertical"
            margin={{ left: 30, right: 30, top: 5, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
              stroke="rgba(255,255,255,0.12)"
            />
            <XAxis type="number" hide />
            <YAxis
              dataKey={dataKeyName}
              type="category"
              width={120}
              tick={{ fontSize: 11, fill: "#FFFFFF" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,0.06)" }}
              contentStyle={{
                borderRadius: "8px",
                border: "none",
                backgroundColor: "#0f172a",
                boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
              }}
              formatter={(value: number) => [value.toLocaleString(), "Audios"]}
            />
            <Bar dataKey={dataKeyValue} fill={color} radius={[0, 4, 4, 0]} barSize={20}>
              {sortedData.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={color}
                  fillOpacity={1 - Math.min(index * 0.05, 0.6)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}