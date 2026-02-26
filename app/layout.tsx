import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignalBoard — ML-Driven Equity Research",
  description:
    "Quantitative signal analysis and ML-derived research reports for equity markets",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-[var(--color-border)] px-6 py-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <a href="/" className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
                SB
              </div>
              <h1 className="text-xl font-semibold tracking-tight">
                SignalBoard
              </h1>
            </a>
            <nav className="flex items-center gap-6 text-sm text-[var(--color-text-dim)]">
              <a href="/" className="hover:text-white transition-colors">
                Dashboard
              </a>
              <span className="text-[var(--color-border)]">|</span>
              <span>ML Research Reports</span>
            </nav>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
