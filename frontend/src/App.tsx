import React, { useState } from 'react';
import { 
  Activity, 
  ShieldCheck, 
  Zap, 
  BarChart3, 
  AlertTriangle,
  RefreshCcw,
  Play
} from 'lucide-react';
import { mlService } from './api/mlService';
import { PredictionResult, DriftReport } from './types';

const App: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);

  const handleTestInference = async (customerId: string) => {
    setLoading(true);
    try {
      const res = await mlService.predict('credit-risk', customerId);
      setPrediction(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCheckDrift = async () => {
    try {
      const res = await mlService.getDrift('credit-risk');
      setDrift(res);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen flex bg-slate-950 font-sans">
      {/* Sidebar */}
      <aside className="w-64 border-r border-slate-800 bg-slate-900/50 p-6 flex flex-col gap-8">
        <div className="flex items-center gap-3 text-orange-500 font-bold text-xl tracking-tight">
          <Zap size={32} fill="currentColor" />
          <span>PHOENIX ML</span>
        </div>
        
        <nav className="flex flex-col gap-2">
          <NavItem icon={<BarChart3 size={20} />} label="Model Registry" active />
          <NavItem icon={<Activity size={20} />} label="Live Traffic" />
          <NavItem icon={<AlertTriangle size={20} />} label="Monitoring" />
          <NavItem icon={<ShieldCheck size={20} />} label="Safety Gates" />
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 overflow-y-auto">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold text-slate-100">Production Dashboard</h1>
          <div className="flex gap-4">
            <button 
              onClick={handleCheckDrift}
              className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded-lg transition-all border border-slate-700"
            >
              <RefreshCcw size={18} className={loading ? "animate-spin" : ""} />
              Scan Drift
            </button>
          </div>
        </header>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <StatCard title="Total Predictions" value="12.5k" subValue="+12% today" color="blue" />
          <StatCard title="Avg Latency" value="45.2ms" subValue="p99: 52.1ms" color="orange" />
          <StatCard title="Active Models" value="2" subValue="v1: Champion, v2: Chall." color="green" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Real-time Test Section */}
          <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-semibold text-slate-200">Interactive Test (A/B Test)</h2>
              <Play size={20} className="text-slate-500" />
            </div>
            
            <div className="flex gap-4 mb-8">
              <TestButton onClick={() => handleTestInference('customer-good')} label="Customer Good" type="good" />
              <TestButton onClick={() => handleTestInference('customer-bad')} label="Customer Bad" type="bad" />
            </div>

            {prediction && (
              <div className="bg-slate-950 rounded-lg p-6 border border-slate-800 animate-in fade-in slide-in-from-bottom-2">
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-widest font-bold mb-1">Assigned Version</p>
                    <span className="text-orange-400 font-mono text-lg">{prediction.version}</span>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-widest font-bold mb-1">Confidence</p>
                    <span className="text-emerald-400 font-mono text-lg">{(prediction.confidence * 100).toFixed(2)}%</span>
                  </div>
                </div>
                <div className="pt-4 border-t border-slate-800">
                  <p className="text-xs text-slate-500 uppercase tracking-widest font-bold mb-2">Prediction Result</p>
                  <div className={`text-3xl font-bold ${prediction.result === 0 ? "text-emerald-500" : "text-rose-500"}`}>
                    {prediction.result === 0 ? "LOW RISK ✅" : "HIGH RISK 🚨"}
                  </div>
                </div>
              </div>
            )}
          </section>

          {/* Monitoring Status */}
          <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <h2 className="text-lg font-semibold text-slate-200 mb-6">Self-Healing Status</h2>
            {drift ? (
              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-slate-950 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-full ${drift.drift_detected ? "bg-rose-500/20 text-rose-500" : "bg-emerald-500/20 text-emerald-500"}`}>
                      <AlertTriangle size={24} />
                    </div>
                    <div>
                      <p className="text-slate-200 font-medium">Data Drift: {drift.feature_name}</p>
                      <p className="text-xs text-slate-500">KS Statistic: {drift.statistic.toFixed(4)}</p>
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${drift.drift_detected ? "bg-rose-500 text-white" : "bg-emerald-500 text-white"}`}>
                    {drift.drift_detected ? "DRIFTED" : "STABLE"}
                  </span>
                </div>
                
                <div className="bg-slate-800/30 p-4 rounded-lg">
                  <p className="text-sm text-slate-400 leading-relaxed italic">
                    {drift.drift_detected 
                      ? "🚨 Recommendation: Monitoring service has detected distribution shift. Automated retraining pipeline has been triggered."
                      : "✅ System is healthy. Model performance is within expected statistical bounds."}
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-slate-600">
                <BarChart3 size={48} className="mb-4 opacity-20" />
                <p>Run a drift scan to see real-time analysis</p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
};

// --- Sub-components (SOLID: Small, reusable) ---

const NavItem: React.FC<{ icon: React.ReactNode, label: string, active?: boolean }> = ({ icon, label, active }) => (
  <div className={`flex items-center gap-3 px-4 py-3 rounded-lg cursor-pointer transition-colors ${active ? "bg-orange-500/10 text-orange-500 border border-orange-500/20" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"}`}>
    {icon}
    <span className="font-medium">{label}</span>
  </div>
);

const StatCard: React.FC<{ title: string, value: string, subValue: string, color: 'blue' | 'orange' | 'green' }> = ({ title, value, subValue, color }) => {
  const colorMap = {
    blue: "border-blue-500/20 bg-blue-500/5",
    orange: "border-orange-500/20 bg-orange-500/5",
    green: "border-emerald-500/20 bg-emerald-500/5",
  };
  const textMap = {
    blue: "text-blue-400",
    orange: "text-orange-400",
    green: "text-emerald-400",
  };
  
  return (
    <div className={`p-6 rounded-xl border ${colorMap[color]} shadow-lg`}>
      <p className="text-slate-500 text-sm font-medium mb-2 uppercase tracking-wide">{title}</p>
      <div className="flex items-end gap-3">
        <span className={`text-3xl font-bold ${textMap[color]}`}>{value}</span>
        <span className="text-xs text-slate-600 mb-1 font-mono">{subValue}</span>
      </div>
    </div>
  );
};

const TestButton: React.FC<{ onClick: () => void, label: string, type: 'good' | 'bad' }> = ({ onClick, label, type }) => (
  <button 
    onClick={onClick}
    className={`flex-1 py-3 rounded-lg font-bold text-sm transition-all shadow-md active:scale-95 ${type === 'good' ? "bg-emerald-600 hover:bg-emerald-500 text-white" : "bg-rose-600 hover:bg-rose-500 text-white"}`}
  >
    Simulate {label}
  </button>
);

export default App;