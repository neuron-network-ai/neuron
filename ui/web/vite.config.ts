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
// DEV ONLY (`apply: 'serve'`), and it never reaches a build. `npm run dev` serves the React app
// with no `ui/app.py` behind it, so every proxied endpoint 404s and the UI renders as a
// signed-out machine with no node — which means the parts that only appear for a CONTRIBUTOR
// (the wallet rows, and the [P39] claim panel) could not be looked at at all without running an
// agent, a coordinator and an OAuth round trip. That is why the claim panel had never been
// clicked by anyone.
//
// These are canned answers in the same shapes `ui/app.py` returns, nothing more. The signature
// itself still needs a real wallet: MetaMask injects `window.ethereum`, and no dev server can.
const devMocks = {
  name: 'neuron-dev-mocks',
  apply: 'serve' as const,
  configureServer(server: { middlewares: { use: (fn: any) => void } }) {
    const canned: Record<string, unknown> = {
      '/node/owner': { is_node: true, needs_owner: true, node_id: 'node-c-pavilion' },
      '/node/payout/challenge': {
        message: 'neuron node:node-c-pavilion binds 0xAbC…  nonce=n-1',
        nonce: 'n-1', address: '0xAbC0000000000000000000000000000000000001',
      },
      '/node/payout/bind': {
        ok: true, payout_address: '0xAbC0000000000000000000000000000000000001',
      },
      '/wallet/balance': {
        logged_in: true, wallet_id: 'w_dev', email: 'dev@example.com',
        balance: 12.5, total_earned: 213.4954,
      },
      '/network': {
        reachable: true, online_nodes: 2, healthy: true, local_capable: false,
        model_id: 'Qwen/Qwen2.5-1.5B-Instruct',
      },
    };
    server.middlewares.use((req: any, res: any, next: any) => {
      const path = String(req.url || '').split('?')[0];
      if (!(path in canned)) return next();
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(canned[path]));
    });
  },
};

export default defineConfig({
  plugins: [react(), tailwindcss(), devMocks],
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
