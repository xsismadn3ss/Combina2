import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    const backend = process.env.BACKEND_INTERNAL_URL || "http://localhost:8000"
    return [
      { source: "/api/v1/:path*", destination: `${backend}/api/v1/:path*` },
      { source: "/health", destination: `${backend}/health` },
      { source: "/ready", destination: `${backend}/ready` },
    ]
  },
}

export default nextConfig
