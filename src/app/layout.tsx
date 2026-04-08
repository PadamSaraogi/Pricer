import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

import Head from "next/head";

export const metadata: Metadata = {
  title: "Pricer | Option & Fixed-Income Analytics",
  description: "Advanced financial analytics for options, bonds, and yield curves.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <link rel="preload" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.js" as="script" />
        <link rel="preload" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.75.0/build/stlite.css" as="style" />
      </head>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
