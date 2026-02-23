import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WorkOrder Risk Dashboard",
  description:
    "GNN-powered risk analysis for SDG&E work order schedules",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 antialiased">{children}</body>
    </html>
  );
}
