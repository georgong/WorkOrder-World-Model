"use client";

import { useHeaderActions } from "@/lib/header-context";

export default function HeaderActions() {
  const { isDemo, hasResult, onReset } = useHeaderActions();

  return (
    <div className="flex items-center gap-2">
      {hasResult && isDemo && (
        <span className="text-[10px] font-medium text-amber-300 bg-amber-900/30 px-2 py-0.5 rounded">
          Demo Mode
        </span>
      )}
      {hasResult && (
        <button
          onClick={onReset}
          className="text-[10px] font-semibold text-blue-200 hover:text-white bg-brand-dark/30 px-3 py-1 rounded transition-colors"
        >
          ← New Analysis
        </button>
      )}
    </div>
  );
}
