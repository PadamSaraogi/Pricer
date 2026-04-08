"use client";

import React, { useState, useEffect } from "react";
import { Calculator } from "lucide-react";

const Card = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => (
  <div className={`glass p-6 ${className}`}>
    {children}
  </div>
);

const Input = ({ label, type = "number", value, onChange, step = "0.01" }: any) => (
  <div className="flex flex-col gap-2 mb-4">
    <label className="text-sm text-secondary font-medium tracking-wide uppercase">{label}</label>
    <input 
      type={type} 
      value={value} 
      onChange={(e) => onChange(e.target.value)}
      step={step}
      className="bg-[#112240] border border-[#233554] rounded-md px-4 py-2 focus:border-accent focus:outline-none transition-colors"
    />
  </div>
);

export const EuropeanPricer = () => {
  const [S0, setS0] = useState(100);
  const [K, setK] = useState(105);
  const [r, setR] = useState(0.05);
  const [sigma, setSigma] = useState(0.20);
  const [T, setT] = useState(0.5);
  const [q, setQ] = useState(0.0);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const calculate = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/pricing/european", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ S0, K, r, sigma, T, q }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      calculate();
    }, 300);
    return () => clearTimeout(timer);
  }, [S0, K, r, sigma, T, q]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-fade-in">
      <Card className="lg:col-span-1">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <Calculator className="w-5 h-5 text-accent" /> Parameters
        </h3>
        <Input label="Spot Price (S₀)" value={S0} onChange={setS0} />
        <Input label="Strike Price (K)" value={K} onChange={setK} />
        <Input label="Risk-free Rate (r)" value={r} onChange={setR} step="0.001" />
        <Input label="Volatility (σ)" value={sigma} onChange={setSigma} step="0.01" />
        <Input label="Time to Expiry (T)" value={T} onChange={setT} />
        <Input label="Dividend Yield (q)" value={q} onChange={setQ} step="0.001" />
      </Card>

      <div className="lg:col-span-2 flex flex-col gap-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="border-l-4 border-accent">
            <p className="text-secondary text-sm uppercase mb-1">Call Price</p>
            <h2 className="text-3xl font-bold">{result?.call_price?.toFixed(4) || "0.0000"}</h2>
          </Card>
          <Card className="border-l-4 border-pink-500">
            <p className="text-secondary text-sm uppercase mb-1">Put Price</p>
            <h2 className="text-3xl font-bold">{result?.put_price?.toFixed(4) || "0.0000"}</h2>
          </Card>
        </div>

        <Card>
          <h3 className="text-lg font-bold mb-4">Greeks</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div>
              <p className="text-secondary text-xs uppercase mb-1">Delta (C)</p>
              <p className="text-xl font-mono">{result?.greeks?.delta?.call?.toFixed(4) || "-"}</p>
            </div>
            <div>
              <p className="text-secondary text-xs uppercase mb-1">Gamma</p>
              <p className="text-xl font-mono">{result?.greeks?.gamma?.toFixed(4) || "-"}</p>
            </div>
            <div>
              <p className="text-secondary text-xs uppercase mb-1">Vega (1%)</p>
              <p className="text-xl font-mono">{result?.greeks?.vega_per_1pct?.toFixed(4) || "-"}</p>
            </div>
            <div>
              <p className="text-secondary text-xs uppercase mb-1">Theta (Day)</p>
              <p className="text-xl font-mono">{result?.greeks?.theta_per_day?.call?.toFixed(4) || "-"}</p>
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-bold mb-4">Moneyness</h3>
          <p className="text-xl font-medium text-accent">{result?.tags?.tag || "Calculating..."}</p>
          <div className="flex gap-4 mt-2">
            <span className="text-sm text-secondary">S/K: {result?.tags?.S_over_K}</span>
            <span className="text-sm text-secondary">d₁: {result?.tags?.d1}</span>
          </div>
        </Card>
      </div>
    </div>
  );
};
