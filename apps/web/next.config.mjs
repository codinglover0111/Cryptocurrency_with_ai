/** @type {import("next").NextConfig} */
const nextConfig = {
  experimental: {
    reactCompiler: true,
    serverActions: {
      bodySizeLimit: "1mb"
    }
  },
  eslint: {
    dirs: ["src"]
  }
};

export default nextConfig;
