import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Relative asset URLs: the same build is served at "/" and under "/topology/" (eisensoftware platform).
  base: "./",
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": process.env.TOPOLOGY_API ?? "http://127.0.0.1:8000" } },
  worker: { format: "es" },
  build: { chunkSizeWarningLimit: 2000 },
});
