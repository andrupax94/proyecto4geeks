import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const isDev = process.env.NODE_ENV === "development";

    const apiBase = isDev
      ? process.env.LOCALAPI || "http://127.0.0.1:8000"
      : process.env.HTTPAPI || "http://andreseduardo.ddns.net:8000";

    return [
      {
        source: "/api/:path*",
        destination: `${apiBase}/:path*`,
      },
    ];
  },
};

export default nextConfig;
