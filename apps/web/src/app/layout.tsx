import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Crypto AI Control Center",
  description: "Monitor the AI-driven trading agent in real time"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
