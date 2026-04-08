"use client";

import React, { useState, useEffect } from "react";
import { Layers } from "lucide-react";

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

export const AmericanPricer = () => {
  const [S0, setS0] = useState(100);
  const [K, setK] = useState(105);
  const [r, setR] = useState(0.05);
  const [sigma, setSigma] = useState(0.20);
  const [T, setT] = useState(0.5);
  const [q, setQ] = useState(0.0);
  const [steps, setSteps] = useState(200);
  const [optionType, setOptionType] = useState("call");
  
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const calculate = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/pricing/american", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ S0, K, r, sigma, T, q, steps, option_type: optionType }),
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
    }, 400);
    return () => clearTimeout(timer);
  }, [S0, K, r, sigma, T, q, steps, optionType]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-fade-in">
      <Card className="lg:col-span-1">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <Layers className="w-5 h-5 text-accent" /> American Options
        </h3>
        
        <div className="flex gap-2 mb-6 p-1 bg-[#0a192f] rounded-lg">
          <button 
            onClick={() => setOptionType("call")}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${optionType === "call" ? "bg-accent text-[#0a192f]" : "text-secondary hover:text-white"}`}
          >
            CALL
          </button>
          <button 
            onClick={() => setOptionType("put")}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${optionType === "put" ? "bg-pink-500 text-white" : "text-secondary hover:text-white"}`}
          >
            PUT
          </button>
        </div>

        <Input label="Spot Price (S₀)" value={S0} onChange={setS0} />
        <Input label="Strike Price (K)" value={K} onChange={setK} />
        <Input label="Risk-free Rate (r)" value={r} onChange={setR} step="0.001" />
        <Input label="Volatility (σ)" value={sigma} onChange={setSigma} step="0.01" />
        <Input label="Time to Expiry (T)" value={T} onChange={setT} />
        <Input label="Binomial Steps" value={steps} onChange={setSteps} step="10" />
      </Card>

      <div className="lg:col-span-2 flex flex-col gap-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="border-l-4 border-accent">
            <p className="text-secondary text-sm uppercase mb-1">American Price (CRR)</p>
            <h2 className="text-3xl font-bold">{result?.amer_tree?.toFixed(4) || "0.0000"}</h2>
          </Card>
          <Card className="border-l-4 border-secondary">
            <p className="text-secondary text-sm uppercase mb-1">European Price (CRR)</p>
            <h2 className="text-3xl font-bold">{result?.euro_tree?.toFixed(4) || "0.0000"}</h2>
          </Card>
        </div>

        <Card>
          <h3 className="text-lg font-bold mb-4">Calculation Notes</h3>
          <div className="space-y-4">
            <div className="p-4 bg-[#1b2a4a] rounded-lg border border-[#233554]">
               <p className="text-sm text-secondary leading-relaxed">
                 The <span className="text-accent font-medium">Cox-Ross-Rubinstein (CRR)</span> model uses a discrete-time binomial tree. 
                 American options can be exercised early, making them more valuable than European counterparts in certain scenarios (e.g., puts with high interest rates or calls with dividends).
               </p>
            </div>
            
            <div className="flex justify-between items-center text-sm border-t border-[#233554] pt-4">
              <span className="text-secondary">Early Exercise Premium:</span>
              <span className="font-mono text-accent">
                {result ? (result.amer_tree - result.euro_tree).toFixed(6) : "0.000000"}
              </span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
