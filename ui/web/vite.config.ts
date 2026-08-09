import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig } from 'vite';

// Builds the chat UI into ui/static/app/, which ui/app.py serves as plain files.
//
// Node is a BUILD dependency only. The agent ships as one PyInstaller bundle and a volunteer
// must never need a JavaScript runtime to use it -- so the deliverable here is static assets,
// exactly like ui/static/chat.html is today.
//
// base is relative on purpose: the bundle is served from a sub-path, and absolute /assets/...
// URLs would 404 there.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: './',
  resolve: {
    alias: { '@': path.resolve(__dirname, '.') },
  },
  build: {
    outDir: path.resolve(__dirname, '../static/app'),
    emptyOutDir: true,
    // The old build warned at 826 kB in one chunk (recharts, framer-motion, highlight.js).
    // Splitting the heavy, rarely-changing libraries out means a returning user re-downloads
    // only the app code -- and this is served off a volunteer's own machine, so the first load
    // is local and cheap while the app chunk is the one that changes every release.
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          markdown: ['highlight.js'],
        },
      },
    },
  },
});
