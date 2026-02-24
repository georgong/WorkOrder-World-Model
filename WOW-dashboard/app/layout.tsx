import type { Metadata } from "next";
import "./globals.css";
import { HeaderProvider } from "@/lib/header-context";
import HeaderActions from "@/components/HeaderActions";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 antialiased">
        <HeaderProvider>
          {/* Global Header */}
          <header className="bg-brand-blue sticky top-0 z-50 shadow-md">
            <div className="max-w-[1440px] mx-auto px-4 py-2 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded bg-brand-green flex items-center justify-center text-white font-bold text-sm">
                    W
                  </div>
                  <div>
                    <h1 className="text-sm font-bold text-white tracking-tight">
                      WorkOrder Scheduling Intelligence Dashboard
                    </h1>
                    <p className="text-[10px] text-blue-200">
                      Graph Neural Network-based Risk Analysis for Work Order Schedules
                    </p>
                  </div>
                </div>
                <HeaderActions />
              </div>
            </div>
          </header>

          {children}
        </HeaderProvider>
      </body>
    </html>
  );
}
