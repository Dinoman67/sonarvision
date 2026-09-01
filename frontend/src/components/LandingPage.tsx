import React, { useEffect } from 'react';
import { Waves, Brain, Globe, Shield, ChevronRight, ArrowRight, Zap } from 'lucide-react';
import coralReef from '../assets/coral-reef.jpg';

interface LandingPageProps {
  onLaunch: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onLaunch }) => {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const delay = entry.target.getAttribute('data-delay');
            if (delay) {
              setTimeout(() => entry.target.classList.add('sr-visible'), parseInt(delay, 10));
            } else {
              entry.target.classList.add('sr-visible');
            }
          }
        });
      },
      { threshold: 0.08, rootMargin: '0px 0px -30px 0px' }
    );

    document.querySelectorAll('.sr').forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen text-slate-200 flex flex-col overflow-hidden" style={{ backgroundColor: '#0b1932' }}>

      {/* ─── FIXED CORAL REEF BACKGROUND ─── */}
      {/* This div is position:fixed, covers the full viewport, and NEVER moves */}
      <div
        className="fixed inset-0 z-0"
        style={{
          backgroundImage: `url(${coralReef})`,
          backgroundSize: '100% 100%',
          backgroundPosition: 'center center',
          backgroundRepeat: 'no-repeat',
          filter: 'brightness(0.3) saturate(0.55)',
          pointerEvents: 'none',
        }}
      />
      {/* Dark navy overlay for text readability */}
      <div
        className="fixed inset-0 z-0"
        style={{
          background: 'rgba(11, 25, 50, 0.45)',
          pointerEvents: 'none',
        }}
      />

      {/* ─── NAVIGATION BAR ─── */}
      <nav className="relative z-50 bg-[#0b1932]/70 backdrop-blur-md border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm font-bold tracking-[0.25em] text-white/90 font-mono-tech uppercase">
              Sonar Vision
            </span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm text-slate-400">
            <a href="#about" className="hover:text-white transition-colors duration-200">About</a>
            <a href="#features" className="hover:text-white transition-colors duration-200">Features</a>
            <a href="#data" className="hover:text-white transition-colors duration-200">Data</a>
            <a href="#contact" className="hover:text-white transition-colors duration-200">Contact</a>
            <div className="h-6 w-px bg-slate-700" />
            <div className="flex items-center gap-1.5 text-emerald-400">
              <Zap className="h-3.5 w-3.5" />
              <span className="font-mono text-xs font-semibold">SIH 2026</span>
            </div>
          </div>
        </div>
      </nav>

      {/* ─── HERO SECTION ─── */}
      <section className="relative z-10 pt-32 pb-20 px-6 min-h-[85vh] flex items-center">
        <div className="max-w-7xl mx-auto w-full">
          <div className="max-w-3xl">
            <h1 className="text-5xl md:text-7xl font-bold leading-tight mb-6 tracking-tight hero-title-animate">
              <span className="text-white">SONAR </span>
              <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">VISION</span>
            </h1>
            <p className="text-lg md:text-xl text-slate-400 leading-relaxed mb-10 max-w-xl hero-subtitle-animate">
              AI-powered marine debris detection from side-scan sonar imagery.
            </p>
            <div className="flex flex-wrap items-center gap-4 hero-buttons-animate">
              <button
                type="button"
                onClick={onLaunch}
                className="group px-7 py-3.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold text-sm flex items-center gap-2 hover:shadow-[0_0_30px_rgba(6,182,212,0.4)] transition-all duration-300 ease-out"
              >
                Launch Detection Tool
                <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform duration-300" />
              </button>
              <a
                href="#features"
                className="px-7 py-3.5 rounded-lg border border-slate-700 text-slate-300 font-semibold text-sm hover:bg-slate-800/50 hover:border-slate-600 transition-all duration-300 ease-out"
              >
                Learn More
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* ─── STATS BAR ─── */}
      <section className="relative z-10 border-y border-white/5 bg-[#0d1f3c]/60 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-6 grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { label: 'mAP50 Score', value: '88.4%', color: 'text-cyan-400', delay: 0 },
            { label: 'Model Size', value: '6.2 MB', color: 'text-emerald-400', delay: 100 },
            { label: 'Parameters', value: '3.3M', color: 'text-amber-400', delay: 200 },
            { label: 'Recall', value: '98.4%', color: 'text-blue-400', delay: 300 },
          ].map((stat) => (
            <div key={stat.label} className="text-center sr" data-delay={stat.delay}>
              <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
              <p className="text-xs text-slate-500 mt-1 font-mono uppercase tracking-wider">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ─── FEATURES SECTION ─── */}
      <section id="features" className="relative z-10 py-20 px-6 bg-[#0b1932]/70 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 sr" data-delay="0">
            <h2 className="text-3xl font-bold text-white mb-4">
              AI-Powered Analysis:{' '}
              <span className="text-cyan-400">Real-Time Monitoring</span>
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Deep learning models for continuous surveillance of marine environments. Precise detection of underwater debris using side-scan sonar imagery.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { icon: Brain, title: 'Spatial-Aware Detection', desc: 'YOLOv8-ESI with SE attention learns shadow patterns and intensity gradients — not just bright spots — for accurate sonar debris identification.' },
              { icon: Globe, title: 'Global Data Insights', desc: 'Mapping pollution trends worldwide with georeferenced detection results. NOAA H11833 benchmark dataset with 834 unseen test images.' },
              { icon: Shield, title: 'Edge Deployment', desc: '6.2 MB ONNX FP16 model runs on Raspberry Pi at 15+ FPS. Lightweight enough for real-time ocean surveying on edge devices.' },
            ].map((f, i) => (
              <div
                key={f.title}
                className="group p-6 rounded-xl bg-[#0d1f3c]/60 border border-white/5 hover:border-cyan-500/30 transition-all duration-300 hover:shadow-[0_0_40px_rgba(6,182,212,0.08)] sr"
                data-delay={String(i * 120)}
              >
                <div className="h-12 w-12 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4 group-hover:bg-cyan-500/20 transition-colors duration-300">
                  <f.icon className="h-6 w-6 text-cyan-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{f.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── ARCHITECTURE SECTION ─── */}
      <section id="data" className="relative z-10 py-20 px-6 bg-[#0a1528]/60 backdrop-blur-sm border-y border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="sr" data-delay="0">
              <h2 className="text-3xl font-bold text-white mb-4">Two-Stage Training Pipeline</h2>
              <p className="text-slate-400 leading-relaxed mb-6">
                Systematic model selection with unseen test validation. YOLOv8-ESI achieves +12.3% mAP50 improvement over baseline YOLOv8n with only 10% more parameters.
              </p>
              <div className="space-y-3">
                {[
                  'Squeeze-and-Excitation attention in C2f backbone',
                  '3.3M parameters — deployable on Raspberry Pi 3',
                  'Noise augmentation: speckle, nadir, acoustic shadows',
                  'FP16/INT8 quantization with accuracy validation',
                ].map((item, i) => (
                  <div key={item} className="flex items-start gap-3 sr" data-delay={String(100 + i * 80)}>
                    <ChevronRight className="h-4 w-4 text-cyan-400 mt-0.5 shrink-0" />
                    <span className="text-sm text-slate-300">{item}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="bg-[#0d1f3c]/60 border border-white/5 rounded-xl p-6 sr" data-delay="200">
              <h3 className="text-sm font-bold text-white mb-4 font-mono-tech uppercase tracking-wider">Model Comparison</h3>
              <div className="space-y-3">
                {[
                  { model: 'YOLOv8n (baseline)', params: '3.01M', map: '78.7%', size: '12 MB' },
                  { model: 'SS-YOLO', params: '1.66M', map: '68.9%', size: '7 MB' },
                  { model: 'YOLOv8-ESI (ours)', params: '3.3M', map: '88.4%', size: '6 MB', highlight: true },
                ].map((m, i) => (
                  <div
                    key={m.model}
                    className={`flex items-center justify-between p-3 rounded border sr ${m.highlight ? 'bg-cyan-500/10 border-cyan-500/30' : 'bg-slate-900/40 border-white/5'}`}
                    data-delay={String(300 + i * 100)}
                  >
                    <span className={`text-sm font-semibold ${m.highlight ? 'text-cyan-300' : 'text-slate-300'}`}>{m.model}</span>
                    <div className="flex items-center gap-4 text-xs font-mono">
                      <span className="text-slate-500">{m.params}</span>
                      <span className={m.highlight ? 'text-cyan-400 font-bold' : 'text-slate-400'}>{m.map}</span>
                      <span className="text-slate-500">{m.size}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─── CTA SECTION ─── */}
      <section className="relative z-10 py-20 px-6 bg-[#0b1932]/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto text-center sr" data-delay="0">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-mono font-semibold mb-6">
            <Waves className="h-3.5 w-3.5" />
            Ready to Analyze
          </div>
          <h2 className="text-3xl font-bold text-white mb-4">Start Detecting Marine Debris</h2>
          <p className="text-slate-400 max-w-xl mx-auto mb-8">
            Upload side-scan sonar imagery or select from curated mission datasets. Real-time AI inference with georeferenced results.
          </p>
          <button
            type="button"
            onClick={onLaunch}
            className="group px-8 py-4 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold text-base flex items-center gap-2 mx-auto hover:shadow-[0_0_40px_rgba(6,182,212,0.5)] transition-all duration-300 ease-out"
          >
            Launch Detection Tool
            <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform duration-300" />
          </button>
        </div>
      </section>

      {/* ─── FOOTER ─── */}
      <footer id="contact" className="relative z-10 border-t border-white/5 bg-[#070f20]/90 backdrop-blur-sm px-6 py-8">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500 font-mono">Sonar Vision — YOLO-ESI Debris Intelligence v1.0</span>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-600 font-mono">
            <span>Smart India Hackathon 2026</span>
            <span>•</span>
            <span>MIT License</span>
            <span>•</span>
            <a href="https://github.com/Dinoman67/sonarvision" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-cyan-400 transition-colors duration-200">
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};
