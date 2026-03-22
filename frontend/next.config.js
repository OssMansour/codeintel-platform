/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_AGENT_API_URL: "/api",
    NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME || "CodeIntel",
  },
  async rewrites() {
    // Proxy /api/* requests through the Next.js server to the agent service.
    // This avoids NEXT_PUBLIC_* env vars baking an unresolvable Docker
    // hostname ("agent:8001") into the client-side JavaScript bundle.
    const agentUrl = process.env.AGENT_INTERNAL_URL || "http://agent:8001";
    return [
      {
        source: "/api/:path*",
        destination: `${agentUrl}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
