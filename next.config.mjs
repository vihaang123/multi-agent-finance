/** @type {import('next').NextConfig} */

// On Vercel the Python function (api/index.py) serves every /api/* request: the rewrite below sends them there.
// For local development / local production tests, set LOCAL_API_URL (e.g. http://127.0.0.1:8000) to proxy to a
// locally running `uvicorn api.index:app` instead.
const localApi = process.env.LOCAL_API_URL;

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: localApi ? `${localApi}/api/:path*` : "/api/",
      },
    ];
  },
};

export default nextConfig;
