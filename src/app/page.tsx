"use client";

import React, { useState } from "react";
import { 
  BarChart3, 
  TrendingUp, 
  Layers, 
  Waves, 
  Percent, 
  Calculator,
  ChevronRight,
  Menu,
  X
} from "lucide-react";
import { EuropeanPricer } from "@/components/EuropeanPricer";
import { AmericanPricer } from "@/components/AmericanPricer";
import { BondCalculator } from "@/components/BondCalculator";

const UnderConstruction = ({ title }: { title: string }) => (
  <div className="glass flex flex-col items-center justify-center py-20 animate-fade-in">
    <Calculator className="w-16 h-16 text-secondary mb-4 opacity-20" />
    <h2 className="text-2xl font-bold mb-2">{title}</h2>
    <p className="text-secondary">This module is currently being migrated.</p>
  </div>
);

export default function PricerDashboard() {
  const [activeTab, setActiveTab] = useState("European");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const tabs = [
    { id: "European", label: "Option Pricer", icon: Calculator, component: EuropeanPricer },
    { id: "American", label: "American (CRR)", icon: Layers, component: AmericanPricer },
    { id: "Bonds", label: "Bond Analytics", icon: BarChart3, component: BondCalculator },
    { id: "Yield", label: "Yield Curve", icon: TrendingUp },
    { id: "Swaps", icon: Waves, label: "Swaps" },
    { id: "Futures", icon: Percent, label: "Black-76" },
  ];

  const ActiveComponent = tabs.find(t => t.id === activeTab)?.component || null;

  return (
    <div className="min-h-screen flex bg-[#0a192f]">
      {/* Sidebar */}
      <aside className={`fixed lg:static z-50 h-full bg-[#112240] border-r border-[#233554] transition-all duration-300 ${isSidebarOpen ? 'w-64' : 'w-0 lg:w-20 overflow-hidden'}`}>
        <div className="p-6 flex items-center gap-3">
          <div className="bg-accent w-8 h-8 rounded-lg flex items-center justify-center">
            <TrendingUp className="text-[#0a192f] w-5 h-5" />
          </div>
          {isSidebarOpen && <h1 className="text-xl font-bold tracking-tight">PRICER</h1>}
        </div>

        <nav className="mt-6 px-3">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all ${
                activeTab === tab.id 
                  ? "bg-[#233554] text-accent font-medium shadow-lg" 
                  : "text-secondary hover:bg-[#1b2a4a] hover:text-white"
              }`}
            >
              <tab.icon className="w-5 h-5 flex-shrink-0" />
              {isSidebarOpen && <span>{tab.label}</span>}
              {activeTab === tab.id && isSidebarOpen && <ChevronRight className="ml-auto w-4 h-4" />}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6 lg:p-10 overflow-y-auto">
        <header className="flex items-center justify-between mb-10">
          <div>
            <h2 className="text-3xl font-bold mb-2">
              {tabs.find(t => t.id === activeTab)?.label}
            </h2>
            <p className="text-secondary">Institutional-grade financial modeling and derivatives pricing.</p>
          </div>
          
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="lg:hidden p-2 text-secondary hover:text-white"
          >
            {isSidebarOpen ? <X /> : <Menu />}
          </button>
        </header>

        <section>
          {ActiveComponent ? <ActiveComponent /> : <UnderConstruction title={tabs.find(t => t.id === activeTab)?.label || ""} />}
        </section>

        <footer className="mt-20 py-6 border-t border-[#233554] flex justify-between items-center text-secondary text-sm">
          <p>© 2026 Saraogi Group Analytics</p>
          <div className="flex gap-6">
            <a href="#" className="hover:text-accent">Terms</a>
            <a href="#" className="hover:text-accent">Privacy</a>
            <a href="#" className="hover:text-accent">Support</a>
          </div>
        </footer>
      </main>
    </div>
  );
}
