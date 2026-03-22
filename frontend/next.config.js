/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_AGENT_API_URL:
      process.env.NEXT_PUBLIC_AGENT_API_URL || "http://localhost:8001",
    NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME || "CodeIntel",
  },
};

module.exports = nextConfig;
