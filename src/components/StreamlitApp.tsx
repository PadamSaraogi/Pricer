'use client';

import React, { useEffect, useRef, useState } from 'react';
import pythonData from '../python_files.json';

const StreamlitApp: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState('Initializing...');

  useEffect(() => {
    // Register Service Worker for caching
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/service-worker.js')
        .then(() => console.log('Service Worker registered'))
        .catch((err) => console.error('Service Worker registration failed', err));
    }

    if (!mountRef.current) return;

    const scriptId = 'stlite-mountable-script';
    const styleId = 'stlite-mountable-style';

    const handleLoad = () => {
      setStatus('Loading Python runtime...');
      setTimeout(() => {
        if (loading) setStatus('Installing analytics libraries (Numpy, Pandas)...');
      }, 5000);
      
      setTimeout(() => {
        if (loading) setStatus('Booting Streamlit engine...');
      }, 15000);

      initializeStlite();
    };

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

      script.onload = handleLoad;
    } else {
      handleLoad();
    }

    function initializeStlite() {
      // @ts-ignore
      if (window.stlite && mountRef.current) {
        // @ts-ignore
        window.stlite.mount({
          entrypoint: "app.py",
          files: pythonData.files,
          requirements: pythonData.requirements,
        }, mountRef.current).then(() => {
          setLoading(false);
        });
      }
    }
  }, []);

  return (
    <div className="fixed inset-0 z-50 bg-white overflow-hidden">
      {loading && (
        <div className="flex h-screen w-screen flex-col items-center justify-center bg-zinc-50 dark:bg-black">
          <div className="flex flex-col items-center gap-6 max-w-sm px-6 text-center">
            {/* Pulsing Analytics Logo / Icon */}
            <div className="relative">
              <div className="h-20 w-20 animate-pulse rounded-2xl bg-zinc-900 flex items-center justify-center shadow-xl dark:bg-zinc-100">
                <svg className="h-10 w-10 text-white dark:text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>

            <div className="space-y-2 w-full">
              <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 uppercase tracking-widest">Pricer</h1>
              <p className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                {status}
              </p>
              
              {/* Modern Indeterminate Progress Bar */}
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                <div className="h-full w-full origin-left animate-progress-indeterminate bg-zinc-900 dark:bg-zinc-100 rounded-full" />
              </div>
              
              <p className="text-[10px] text-zinc-400 dark:text-zinc-500 italic">
                First load downloads the 30MB Python runtime. Subsequent loads are instant.
              </p>
            </div>
          </div>
        </div>
      )}
      <div ref={mountRef} className="h-full w-full" />
      
      <style jsx global>{`
        @keyframes progress-indeterminate {
          0% { transform: scaleX(0.1) translateX(-10%); }
          50% { transform: scaleX(0.4) translateX(100%); }
          100% { transform: scaleX(0.1) translateX(1000%); }
        }
        .animate-progress-indeterminate {
          animation: progress-indeterminate 2s infinite linear;
        }
      `}</style>
    </div>
  );
};

export default StreamlitApp;
