import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server runs on 5173; the backend API/WebSocket runs on 127.0.0.1:8000.
// The packaged app (M5) serves this build same-origin from FastAPI.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: { outDir: 'dist' },
})
