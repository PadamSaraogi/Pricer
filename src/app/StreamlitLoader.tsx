"use client";

import dynamic from "next/dynamic";

const StreamlitApp = dynamic(() => import("../components/StreamlitApp"), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen w-screen items-center justify-center bg-zinc-50 dark:bg-black">
      <div className="flex flex-col items-center gap-4">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-zinc-300 border-t-zinc-900 dark:border-zinc-700 dark:border-t-zinc-50"></div>
        <p className="text-lg font-medium text-zinc-600 dark:text-zinc-400">
          Initializing Analytics Dashboard...
        </p>
      </div>
    </div>
  ),
});

export default function StreamlitLoader() {
  return <StreamlitApp />;
}
