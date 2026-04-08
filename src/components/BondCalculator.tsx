"use client";

import React, { useState, useEffect } from "react";
import { BarChart3 } from "lucide-react";

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

export const BondCalculator = () => {
  const [face, setFace] = useState(100);
  const [coupon, setCoupon] = useState(0.05);
  const [yieldRate, setYieldRate] = useState(0.04);
  const [T, setT] = useState(5);
  const [freq, setFreq] = useState(2);
  
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const calculate = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/bond/analytics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ face, coupon, yield: yieldRate, T, freq }),
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
  }, [face, coupon, yieldRate, T, freq]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-fade-in">
      <Card className="lg:col-span-1">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-accent" /> Bond Parameters
        </h3>
        
        <Input label="Face Value" value={face} onChange={setFace} step="1" />
        <Input label="Coupon Rate (annual)" value={coupon} onChange={setCoupon} step="0.001" />
        <Input label="Yield to Maturity (YTM)" value={yieldRate} onChange={setYieldRate} step="0.001" />
        <Input label="Time to Maturity (Years)" value={T} onChange={setT} />
        
        <div className="flex flex-col gap-2 mb-4">
          <label className="text-sm text-secondary font-medium tracking-wide uppercase">Payment Frequency</label>
          <select 
            value={freq} 
            onChange={(e) => setFreq(parseInt(e.target.value))}
            className="bg-[#112240] border border-[#233554] rounded-md px-4 py-2 focus:border-accent focus:outline-none transition-colors appearance-none"
          >
            <option value={1}>Annual</option>
            <option value={2}>Semi-Annual</option>
            <option value={4}>Quarterly</option>
            <option value={12}>Monthly</option>
          </select>
        </div>
      </Card>

      <div className="lg:col-span-2 flex flex-col gap-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="border-l-4 border-accent">
            <p className="text-secondary text-sm uppercase mb-1">Present Value (Price)</p>
            <h2 className="text-3xl font-bold">{result?.price?.toFixed(4) || "0.0000"}</h2>
          </Card>
          <Card className="border-l-4 border-secondary">
            <p className="text-secondary text-sm uppercase mb-1">Current Yield</p>
            <h2 className="text-3xl font-bold">{((coupon * face) / (result?.price || face) * 100).toFixed(2)}%</h2>
          </Card>
        </div>

        <Card>
          <h3 className="text-lg font-bold mb-4">Risk Analytics</h3>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <p className="text-secondary text-xs uppercase mb-1">Macaulay Duration</p>
              <p className="text-xl font-mono">{result?.macaulay_duration?.toFixed(4) || "-"}</p>
            </div>
            <div>
              <p className="text-secondary text-xs uppercase mb-1">Modified Duration</p>
              <p className="text-xl font-mono">{result?.modified_duration?.toFixed(4) || "-"}</p>
            </div>
            <div>
              <p className="text-secondary text-xs uppercase mb-1">Convexity</p>
              <p className="text-xl font-mono">{result?.convexity?.toFixed(4) || "-"}</p>
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-bold mb-4">Interpretation</h3>
          <div className="p-4 bg-[#1b2a4a] rounded-lg border border-[#233554]">
             <p className="text-sm text-secondary leading-relaxed">
               A modified duration of <span className="text-accent">{result?.modified_duration?.toFixed(4)}</span> suggests that for every 1% change in yields, the bond price is expected to move by approximately <span className="text-accent">{(result?.modified_duration * 1).toFixed(2)}%</span> in the opposite direction.
             </p>
          </div>
        </Card>
      </div>
    </div>
  );
};
