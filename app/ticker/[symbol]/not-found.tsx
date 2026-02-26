import Link from "next/link";

export default function TickerNotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] text-center">
      <h2 className="text-2xl font-bold mb-2">Ticker Not Found</h2>
      <p className="text-[var(--color-text-dim)] mb-6">
        No ML research report is available for this ticker symbol.
      </p>
      <Link
        href="/"
        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors"
      >
        Back to Dashboard
      </Link>
    </div>
  );
}
