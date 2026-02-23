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
    <main className="min-h-screen">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                WorkOrder Risk Dashboard
              </h1>
              <p className="text-sm text-gray-500 mt-1">
                GNN-powered risk analysis for SDG&E work order schedules
              </p>
            </div>
            {result && (
              <button
                onClick={() => {
                  setResult(null);
                  setError(null);
                }}
                className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
              >
                New Upload
              </button>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
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
          <Dashboard data={result} />
        )}
      </div>
    </main>
  );
}
