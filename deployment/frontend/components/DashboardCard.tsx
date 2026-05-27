interface DashboardCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: string;
  icon?: string;
}

export default function DashboardCard({ title, value, subtitle, color = "blue", icon }: DashboardCardProps) {
  const colorMap: Record<string, string> = {
    blue: "border-blue-500 bg-blue-50 text-blue-700",
    green: "border-green-500 bg-green-50 text-green-700",
    red: "border-red-500 bg-red-50 text-red-700",
    yellow: "border-yellow-500 bg-yellow-50 text-yellow-700",
    purple: "border-purple-500 bg-purple-50 text-purple-700",
    white: "text-gray-100",
  };

  return (
    <div className={`rounded-lg p-5`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium uppercase tracking-wide opacity-70">{title}</span>
        {icon && <span className="text-2xl">{icon}</span>}
      </div>
      <p className="text-3xl font-bold">{value}</p>
      {subtitle && <p className="text-xs mt-1 opacity-60">{subtitle}</p>}
    </div>
  );
}
