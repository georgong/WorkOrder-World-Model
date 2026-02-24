"use client";

import { useEffect, useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import Dashboard from "@/components/Dashboard";
import { fetchDemo } from "@/lib/api";
import { useSetHeader } from "@/lib/header-context";
import type { PredictResponse } from "@/lib/types";

export default function Home() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const setHeader = useSetHeader();

  const resetAll = () => {
    setResult(null);
    setError(null);
  };

  // Keep header in sync with page state
  useEffect(() => {
    setHeader({
      hasResult: !!result,
      isDemo: result?.metadata?.mode === "demo",
      onReset: resetAll,
    });
  }, [result, setHeader]);

  const handleResult = (data: PredictResponse) => {
    setResult(data);
    setError(null);
  };

  const handleDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDemo();
      setResult(data);
    } catch (e: any) {
      setError(e.message ?? "Failed to load demo data");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50/50">
      <div className="max-w-[1440px] mx-auto px-4 py-4 sm:px-6 lg:px-8">
        {/* Error Banner */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex">
              <span className="text-red-500 text-lg mr-2">⚠</span>
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
            onDemo={handleDemo}
            loading={loading}
            setLoading={setLoading}
            setError={setError}
          />
        ) : (
          <Dashboard
            data={result}
            onReset={resetAll}
          />
        )}
      </div>
    </main>
  );
}
