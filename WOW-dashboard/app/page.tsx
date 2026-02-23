"use client";

import { useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import Dashboard from "@/components/Dashboard";
import type { PredictResponse } from "@/lib/types";

export default function Home() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleResult = (data: PredictResponse) => {
    setResult(data);
    setError(null);
  };

  return (
    <main className="min-h-screen bg-slate-50/50">
      {/* Header */}
      <header className="bg-brand-blue sticky top-0 z-50 shadow-md">
        <div className="max-w-[1440px] mx-auto px-4 py-2 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-brand-green flex items-center justify-center text-white font-bold text-sm">
                W
              </div>
              <div>
                <h1 className="text-sm font-bold text-white tracking-tight">
                  WorkOrder Risk Dashboard
                </h1>
                <p className="text-[10px] text-blue-200">
                  SDG&E Optimisation Engine
                </p>
              </div>
            </div>
            {result && (
               <div className="bg-brand-dark/30 px-2 py-0.5 rounded text-[10px] text-blue-100 font-mono">
                  v1.2.0-stable
               </div>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-[1440px] mx-auto px-4 py-4 sm:px-6 lg:px-8">
        {/* Error Banner */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex">
              <span className="text-red-500 text-lg mr-2"></span>
              <div>
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Upload or Dashboard */}
        {!result ? (
          <UploadPanel
            onResult={handleResult}
            loading={loading}
            setLoading={setLoading}
            setError={setError}
          />
        ) : (
          <Dashboard 
            data={result} 
            onReset={() => {
                setResult(null);
                setError(null);
            }} 
          />
        )}
      </div>
    </main>
  );
}
