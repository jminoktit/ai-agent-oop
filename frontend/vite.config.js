import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/login': 'http://127.0.0.1:8000',
      '/register': 'http://127.0.0.1:8000',
      '/logout': 'http://127.0.0.1:8000',
      '/chat': 'http://127.0.0.1:8000',
      '/switch-agent': 'http://127.0.0.1:8000',
      '/new-conversation': 'http://127.0.0.1:8000',
      '/conversation': 'http://127.0.0.1:8000',
      '/agent-info': 'http://127.0.0.1:8000',
      '/conversations': 'http://127.0.0.1:8000',
      '/user-info': 'http://127.0.0.1:8000',
      '/files': 'http://127.0.0.1:8000',
      '/training': 'http://127.0.0.1:8000',
      '/settings': 'http://127.0.0.1:8000',
      '/media': 'http://127.0.0.1:8000',
      '/admin': 'http://127.0.0.1:8000',
    },
  },
});
