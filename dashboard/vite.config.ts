import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/dashboard/',
  build: {
    outDir: 'dist',
  },
  server: {
    // `npm run dev` proxies API calls to the FastAPI engine.
    proxy: {
      '/v1': 'http://localhost:8000',
    },
  },
})
