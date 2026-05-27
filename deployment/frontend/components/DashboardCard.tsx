interface PieSegment {
  value: number;
  color: "blue" | "green" | "red" | "yellow" | "purple";
}

interface DashboardCardProps {
  title: string;
  value: string | number;
  subtitle?: string;

  // Color clásico (modo simple)
  color?: "blue" | "green" | "red" | "yellow" | "purple" | "white";

  // Ahora 'icon' representa la URL de la imagen/SVG para la máscara
  icon?: string;
  box_class?: string
  // Modo clásico
  currentValue?: number;
  maxValue?: number;

  // NUEVO → modo múltiple
  values?: PieSegment[];
}

export default function DashboardCard({
  title,
  value,
  subtitle,
  color = "blue",
  icon,
  currentValue, // Quitamos el default = 0 para poder validar si viene o no
  maxValue = 100,
  box_class = "box box-black",
  values,
}: DashboardCardProps) {

  const radius = 32;
  const circumference = 2 * Math.PI * radius;

  const ringColorMap = {
    blue: "stroke-blue-500 text-blue-400",
    green: "stroke-green-500 text-green-400",
    red: "stroke-red-500 text-red-400",
    yellow: "stroke-yellow-500 text-yellow-400",
    purple: "stroke-purple-500 text-purple-400",
    white: "stroke-gray-100 text-gray-100",
  };

  const strokeColorMap = {
    blue: "#3b82f6",
    green: "#22c55e",
    red: "#ef4444",
    yellow: "#eab308",
    purple: "#a855f7",
    white: "#FFFFFF",
  };

  // Validamos si tenemos datos para mostrar gráficos (Modo Simple o Modo Múltiple)
  const hasChartData = values && values.length > 0 || currentValue !== undefined;

  // =========================================
  // MODO SIMPLE (ACTUAL)
  // =========================================
  const safeCurrentValue = currentValue ?? 0;
  const percentage =
    maxValue > 0
      ? Math.min(Math.round((safeCurrentValue / maxValue) * 100), 100)
      : 0;

  const strokeDashoffset =
    circumference - (percentage / 100) * circumference;

  const selectedRingColor =
    ringColorMap[color] || ringColorMap.blue;

  // =========================================
  // MODO MULTI-PORCENTAJE
  // =========================================
  const total =
    values?.reduce((acc, item) => acc + item.value, 0) || 0;

  let accumulated = 0;

  return (
    <div className={box_class + ` rounded-xl p-5 flex items-center justify-between shadow-sm max-w-sm`}>

      {/* TEXTOS */}
      <div className="flex flex-col justify-between h-full">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-gray-400 block mb-1">
            {title}
          </span>

          <p className="text-3xl font-bold tracking-tight text-gray-100">
            {value}
          </p>
        </div>

        {subtitle && (
          <p className="text-xs mt-4 text-gray-300 font-medium">
            {subtitle}
          </p>
        )}
      </div>

      {/* BLOQUE DERECHO (ANILLO O ICONO MASK) */}
      <div className="relative flex items-center justify-center w-24 h-24 ml-4">

        {hasChartData ? (
          <>
            {/* Si hay datos, renderizamos todo el ecosistema del SVG gráfico */}
            <svg
              className="w-full h-full transform -rotate-90"
              viewBox="0 0 80 80"
            >
              {/* Fondo */}
              <circle
                cx="40"
                cy="40"
                r={radius}
                className="stroke-gray-700"
                strokeWidth="8"
                fill="transparent"
              />

              {/* MODO MULTIPLE */}
              {values && values.length > 0 ? (
                values.map((segment, index) => {
                  const segmentPercentage = total > 0 ? segment.value / total : 0;
                  const dash = circumference * segmentPercentage;
                  const offset = circumference - accumulated;
                  accumulated += dash;

                  return (
                    <circle
                      key={index}
                      cx="40"
                      cy="40"
                      r={radius}
                      stroke={strokeColorMap[segment.color]}
                      strokeWidth="8"
                      fill="transparent"
                      strokeDasharray={`${dash} ${circumference}`}
                      strokeDashoffset={offset}
                      strokeLinecap="round"
                      className="transition-all duration-500"
                    />
                  );
                })
              ) : (
                /* MODO SIMPLE */
                <circle
                  cx="40"
                  cy="40"
                  r={radius}
                  className={`transition-all duration-500 ease-out ${selectedRingColor}`}
                  strokeWidth="8"
                  fill="transparent"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                />
              )}
            </svg>

            {/* TEXTO CENTRAL DEL ANILLO */}
            <div
              className={`absolute inset-0 flex flex-col items-center justify-center ${values?.length ? "text-gray-200" : selectedRingColor
                }`}
            >
              <span className="text-sm font-bold">
                {values?.length ? `${total}` : `${percentage}%`}
              </span>

              {icon && (
                <span
                  className="text-xs opacity-80 mt-0.5 w-4 h-4 bg-current"
                  style={{
                    maskImage: `url(${icon})`,
                    WebkitMaskImage: `url(${icon})`,
                    maskSize: 'contain',
                    WebkitMaskSize: 'contain',
                    maskRepeat: 'no-repeat',
                    WebkitMaskRepeat: 'no-repeat'
                  }}
                />
              )}
            </div>
          </>
        ) : (
          /* MODO SÓLO ICONO MASK (Del tamaño completo del contenedor del anillo 24x24 / w-full h-full) */
          icon && (
            <div
              className={`w-full h-full bg-current transition-all ${selectedRingColor}`}
              style={{
                maskImage: `url(${icon})`,
                maskPosition: `center`,
                WebkitMaskImage: `url(${icon})`,
                width: "60%",
                height: "60%",
                maskSize: 'contain',
                WebkitMaskSize: 'contain',
                maskRepeat: 'no-repeat',
                WebkitMaskRepeat: 'no-repeat'
              }}
            />
          )
        )}
      </div>
    </div>
  );
}