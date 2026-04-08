'use client';

import React, { useEffect, useRef, useState } from 'react';
import pythonData from '../python_files.json';

const StreamlitApp: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!mountRef.current) return;

    // Load stlite-mountable from CDN
    // We use the CDN approach to avoid React 18/19 compatibility issues with the NPM package
    const scriptId = 'stlite-mountable-script';
    const styleId = 'stlite-mountable-style';

    if (!document.getElementById(scriptId)) {
      const script = document.createElement('script');
      script.id = scriptId;
      script.src = "https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.js";
      script.async = true;
      document.head.appendChild(script);

      const style = document.createElement('link');
      style.id = styleId;
      style.rel = "stylesheet";
      style.href = "https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.css";
      document.head.appendChild(style);

      script.onload = () => {
        initializeStlite();
        setLoading(false);
      };
    } else {
      // Script already loaded
      initializeStlite();
      setLoading(false);
    }

    function initializeStlite() {
      // @ts-ignore
      if (window.stlite && mountRef.current) {
        // @ts-ignore
        window.stlite.mount({
          entrypoint: "app.py",
          files: pythonData.files,
          requirements: pythonData.requirements,
        }, mountRef.current);
      }
    }
  }, []);

  return (
    <div className="fixed inset-0 z-50 bg-white overflow-hidden">
      {loading && (
        <div className="flex h-screen w-screen flex-col items-center justify-center bg-zinc-50 dark:bg-black">
          <div className="flex flex-col items-center gap-4">
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-zinc-300 border-t-zinc-900 dark:border-zinc-700 dark:border-t-zinc-50"></div>
            <p className="text-lg font-medium text-zinc-600 dark:text-zinc-400">
              Initializing Analytics Engine...
            </p>
          </div>
        </div>
      )}
      <div ref={mountRef} className="h-full w-full" />
    </div>
  );
};

export default StreamlitApp;
