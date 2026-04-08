import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Pricer | Saraogi Group Analytics",
  description: "Institutional-grade derivative and fixed-income pricing dashboard.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-[#0a192f] text-[#e6f1ff]`}>
        {children}
      </body>
    </html>
  );
}
