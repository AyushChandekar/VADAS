import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendHost = process.env.VITE_BACKEND_HOST || 'localhost'
const backendPort = process.env.VITE_BACKEND_PORT || '8000'
const backendUrl = `http://${backendHost}:${backendPort}`

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: backendUrl,
        changeOrigin: true,
      },
    },
  },
})
