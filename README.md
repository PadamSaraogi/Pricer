# Pricer: Option & Fixed-Income Analytics

This project is a comprehensive analytics dashboard for European/American options, bonds, yield curves, and more. It has been migrated from a standalone Streamlit app to a Vercel-ready Next.js application using **stlite**.

## Migration to Vercel (stlite)

To enable hosting on Vercel's serverless infrastructure without a persistent Python server, this project uses **stlite** (Streamlit on WebAssembly). The entire Streamlit application runs directly in the user's browser.

### How it works
- **Logic**: The original Streamlit logic resides in the `backend/` directory.
- **Bundling**: During build, Python files are bundled into `src/python_files.json`.
- **Frontend**: A Next.js wrapper (`src/app/page.tsx`) loads the **stlite** runtime and mounts the bundled Python files.
- **Compatibility**: The frontend uses a client-side loader with `ssr: false` to ensure Pyodide/stlite only initializes in the browser.

### Local Development
1. **Install dependencies**:
   ```bash
   npm install
   ```
2. **Bundle Python files**:
   ```bash
   node scripts/bundle_python.js
   ```
3. **Start the development server**:
   ```bash
   npm run dev
   ```
4. **Open the browser**:
   Navigate to [http://localhost:3000](http://localhost:3000).

### Deployment on Vercel
1. Push this repository to GitHub.
2. Link the repository to a new project in your Vercel dashboard.
3. Configure the **Build Command** to include the bundling step:
   ```bash
   node scripts/bundle_python.js && next build
   ```
4. Vercel will automatically deploy the app as a static/Next.js site.

## Features
- **Zero Visual Changes**: Maintains the exact same Streamlit interface.
- **Zero Functional Changes**: Uses existing Python mathematical models and dashboards.
- **Serverless Analytics**: No persistent Python backend required.
