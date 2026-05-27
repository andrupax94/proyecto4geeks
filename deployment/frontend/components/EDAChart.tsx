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

interface EDAChartProps {
  data: { class: string; count: number }[];
  title: string;
  color?: string;
}

export default function EDAChart({
  data = [],
  title,
  color = "#3b82f6",
}: EDAChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="box black-box p-5 h-[350px] flex flex-col items-center justify-center">
        <h3 className="text-base font-semibold text-gray-200 mb-4 self-start">
          {title}
        </h3>
        <p className="text-gray-400 text-sm">
          Cargando distribución de clases...
        </p>
      </div>
    );
  }

  const sortedData = [...data].sort((a, b) => b.count - a.count);

  return (
    <div className="box black-box shadow p-5 transition-all duration-200 hover:blue-box">
      <h3 className="text-base font-semibold text-gray-200 mb-4">{title}</h3>

      <div className="box blue-box-no-after p-3 h-[300px] w-full transition-all duration-200 hover:shadow-lg">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sortedData}
            layout="vertical"
            margin={{ left: 20, right: 30, top: 5, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="4 4"
              horizontal={false}
              stroke="rgba(255,255,255,0.12)"
            />
            <XAxis type="number" hide />
            <YAxis
              dataKey="class"
              type="category"
              width={130}
              tick={{ fontSize: 11, fill: "#FFFFFF" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(value) => String(value).replace(/_/g, " ")}
            />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,0.06)" }}
              contentStyle={{
                borderRadius: "8px",
                border: "none",
                backgroundColor: "#0f172a",
                color: "#fff",
                boxShadow: "0 4px 12px rgba(0,0,0,0.25)",
              }}
              formatter={(value: number) => [value.toLocaleString(), "Audios"]}
            />
            <Bar dataKey="count" fill={color} radius={[0, 4, 4, 0]} barSize={18}>
              {sortedData.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={color}
                  fillOpacity={1 - Math.min(index * 0.04, 0.7)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}