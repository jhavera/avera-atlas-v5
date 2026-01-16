import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as THREE from 'three';

// ============================================================================
// AVERA-ATLAS Interactive Technical Demo
// Space Debris Detection & Conjunction Assessment System
// ============================================================================

const COLORS = {
  background: '#0a0e14',
  backgroundAlt: '#0d1117',
  primary: '#00d4ff',
  secondary: '#7c3aed',
  accent: '#f59e0b',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  text: '#e6edf3',
  textMuted: '#8b949e',
  border: '#21262d',
  earth: '#1a4a7a',
  earthGlow: '#3b82f6',
  debris: '#ef4444',
  spacecraft: '#10b981',
  orbit: '#374151',
};

const RISK_COLORS = {
  GREEN: '#10b981',
  AMBER: '#f59e0b',
  RED: '#ef4444',
};

// ============================================================================
// Tracker Algorithm Explainer Component
// Visual explanation of IOD methods for investors and technical team
// ============================================================================

function TrackerExplainer({ isPlaying }) {
  const [activeSection, setActiveSection] = useState(0);
  const [animationPhase, setAnimationPhase] = useState(0);

  const sections = [
    { id: 'problem', title: 'The Challenge', icon: '?' },
    { id: 'correlation', title: 'Cross-Sensor Correlation', icon: '⊕' },
    { id: 'gauss', title: 'Gauss IOD', icon: 'G' },
    { id: 'herrick', title: 'Herrick-Gibbs', icon: 'H' },
    { id: 'covariance', title: 'Uncertainty', icon: 'σ' },
    { id: 'industry', title: 'Industry Standards', icon: '★' },
  ];

  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setAnimationPhase(p => (p + 1) % 100);
    }, 50);
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Animated observation dots for visualizations
  const ObservationDot = ({ x, y, color, delay, label, size = 12 }) => {
    const pulse = Math.sin((animationPhase + delay) * 0.1) * 0.3 + 1;
    return (
      <g>
        <circle
          cx={x}
          cy={y}
          r={size * pulse}
          fill={color}
          opacity={0.3}
        />
        <circle
          cx={x}
          cy={y}
          r={size * 0.6}
          fill={color}
        />
        {label && (
          <text
            x={x}
            y={y + size + 14}
            textAnchor="middle"
            fill="#8b949e"
            fontSize="11"
            fontFamily="IBM Plex Mono"
          >
            {label}
          </text>
        )}
      </g>
    );
  };

  // The Problem Section
  const ProblemSection = () => (
    <div style={{ padding: '32px 40px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32 }}>
        {/* Single Observation Problem */}
        <div style={{
          background: 'rgba(239,68,68,0.05)',
          border: '1px solid rgba(239,68,68,0.2)',
          borderRadius: 16,
          padding: 28,
        }}>
          <h4 style={{ color: '#ef4444', fontSize: 18, marginBottom: 20, letterSpacing: '1px', fontWeight: 600 }}>
            ✗ SINGLE OBSERVATION
          </h4>
          <svg viewBox="0 0 300 200" style={{ width: '100%', height: 180 }}>
            {/* Earth */}
            <circle cx="150" cy="180" r="80" fill="#1a4a7a" opacity="0.3" />
            <ellipse cx="150" cy="180" rx="80" ry="20" fill="none" stroke="#3b82f6" strokeWidth="1" opacity="0.3" />
            
            {/* Single CubeSat */}
            <rect x="60" y="70" width="16" height="12" fill="#00d4ff" rx="2" />
            <rect x="50" y="73" width="10" height="6" fill="#1e3a5f" />
            <rect x="76" y="73" width="10" height="6" fill="#1e3a5f" />
            
            {/* Debris - unknown position */}
            <ObservationDot x={200} y={50} color="#ef4444" delay={0} />
            
            {/* Line of sight - but where on this line? */}
            <line x1="76" y1="76" x2="280" y2="20" stroke="#ef4444" strokeWidth="2" strokeDasharray="5,5" opacity="0.5" />
            
            {/* Question marks along the line */}
            <text x="120" y="55" fill="#ef4444" fontSize="16" fontWeight="bold">?</text>
            <text x="160" y="42" fill="#ef4444" fontSize="16" fontWeight="bold">?</text>
            <text x="200" y="30" fill="#ef4444" fontSize="16" fontWeight="bold">?</text>
            <text x="240" y="18" fill="#ef4444" fontSize="14" fontWeight="bold">?</text>
          </svg>
          <p style={{ fontSize: 15, color: '#8b949e', lineHeight: 1.7, marginTop: 16 }}>
            A single observation gives us <strong style={{ color: '#ef4444' }}>direction only</strong>—we know 
            where to look, but not how far away the object is. The debris could be anywhere along this line of sight.
          </p>
        </div>

        {/* Multi-Observation Solution */}
        <div style={{
          background: 'rgba(16,185,129,0.05)',
          border: '1px solid rgba(16,185,129,0.2)',
          borderRadius: 16,
          padding: 28,
        }}>
          <h4 style={{ color: '#10b981', fontSize: 18, marginBottom: 20, letterSpacing: '1px', fontWeight: 600 }}>
            ✓ THREE OBSERVATIONS
          </h4>
          <svg viewBox="0 0 300 200" style={{ width: '100%', height: 180 }}>
            {/* Earth */}
            <circle cx="150" cy="180" r="80" fill="#1a4a7a" opacity="0.3" />
            
            {/* Three CubeSats */}
            <rect x="40" y="90" width="14" height="10" fill="#00d4ff" rx="2" />
            <rect x="140" y="40" width="14" height="10" fill="#8b5cf6" rx="2" />
            <rect x="240" y="80" width="14" height="10" fill="#10b981" rx="2" />
            
            {/* Debris - determined position */}
            <ObservationDot x={150} y={90} color="#10b981" delay={0} size={10} />
            
            {/* Lines of sight converging */}
            <line x1="54" y1="95" x2="150" y2="90" stroke="#00d4ff" strokeWidth="2" opacity="0.7" />
            <line x1="147" y1="50" x2="150" y2="90" stroke="#8b5cf6" strokeWidth="2" opacity="0.7" />
            <line x1="240" y1="85" x2="150" y2="90" stroke="#10b981" strokeWidth="2" opacity="0.7" />
            
            {/* Convergence indicator */}
            <circle cx="150" cy="90" r="20" fill="none" stroke="#10b981" strokeWidth="1" strokeDasharray="3,3" opacity="0.5" />
          </svg>
          <p style={{ fontSize: 15, color: '#8b949e', lineHeight: 1.7, marginTop: 16 }}>
            Three observations from different positions create <strong style={{ color: '#10b981' }}>intersecting lines of sight</strong>. 
            This triangulation uniquely determines the object's 3D position and enables orbit calculation.
          </p>
        </div>
      </div>

      {/* Key insight box */}
      <div style={{
        marginTop: 28,
        padding: 24,
        background: 'linear-gradient(135deg, rgba(0,212,255,0.1), rgba(124,58,237,0.1))',
        borderRadius: 12,
        border: '1px solid rgba(0,212,255,0.2)',
      }}>
        <div style={{ fontSize: 15, color: '#00d4ff', fontWeight: 600, marginBottom: 10 }}>
          KEY INSIGHT
        </div>
        <p style={{ fontSize: 16, color: '#e6edf3', lineHeight: 1.8, margin: 0 }}>
          With three time-separated observations, we can solve for six orbital elements 
          (a, e, i, Ω, ω, ν) that completely describe the debris trajectory. This is the 
          fundamental principle behind <strong>Initial Orbit Determination (IOD)</strong>.
        </p>
      </div>
    </div>
  );

  // Cross-Sensor Correlation Section
  const CorrelationSection = () => (
    <div style={{ padding: '32px 40px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 40 }}>
        <div>
          <svg viewBox="0 0 400 300" style={{ width: '100%', height: 300 }}>
            {/* Timeline base */}
            <line x1="50" y1="250" x2="380" y2="250" stroke="#374151" strokeWidth="2" />
            <text x="215" y="280" textAnchor="middle" fill="#8b949e" fontSize="12">Time →</text>
            
            {/* CubeSat tracks */}
            {[
              { y: 60, color: '#00d4ff', label: 'ATLAS-1' },
              { y: 130, color: '#8b5cf6', label: 'ATLAS-2' },
              { y: 200, color: '#10b981', label: 'ATLAS-3' },
            ].map((sat, i) => (
              <g key={i}>
                <text x="30" y={sat.y + 5} textAnchor="end" fill={sat.color} fontSize="11" fontFamily="IBM Plex Mono">
                  {sat.label}
                </text>
                <line x1="50" y1={sat.y} x2="380" y2={sat.y} stroke={sat.color} strokeWidth="1" opacity="0.3" />
              </g>
            ))}

            {/* Detection events - showing correlation */}
            {/* Object A detections */}
            <ObservationDot x={100} y={60} color="#ef4444" delay={0} size={8} />
            <ObservationDot x={160} y={130} color="#ef4444" delay={10} size={8} />
            <ObservationDot x={220} y={200} color="#ef4444" delay={20} size={8} />
            
            {/* Correlation lines for Object A */}
            <path 
              d={`M 100 68 Q 130 100 160 130 Q 190 165 220 192`}
              fill="none" 
              stroke="#ef4444" 
              strokeWidth="2" 
              strokeDasharray="4,4"
              opacity={0.6 + Math.sin(animationPhase * 0.1) * 0.3}
            />
            
            {/* Object B detections (different timing) */}
            <ObservationDot x={140} y={60} color="#f59e0b" delay={30} size={8} />
            <ObservationDot x={240} y={130} color="#f59e0b" delay={40} size={8} />
            <ObservationDot x={320} y={200} color="#f59e0b" delay={50} size={8} />
            
            {/* Correlation lines for Object B */}
            <path 
              d={`M 140 68 Q 190 100 240 130 Q 280 165 320 192`}
              fill="none" 
              stroke="#f59e0b" 
              strokeWidth="2" 
              strokeDasharray="4,4"
              opacity={0.6 + Math.sin(animationPhase * 0.1 + 1) * 0.3}
            />

            {/* Labels */}
            <text x="160" y="240" textAnchor="middle" fill="#ef4444" fontSize="11" fontWeight="600">
              DEBRIS-001
            </text>
            <text x="280" y="240" textAnchor="middle" fill="#f59e0b" fontSize="11" fontWeight="600">
              DEBRIS-002
            </text>
          </svg>
        </div>

        <div>
          <h4 style={{ color: '#e6edf3', fontSize: 22, marginBottom: 20, fontWeight: 600 }}>
            Correlation Algorithm
          </h4>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {[
              { step: '1', title: 'Angular Velocity Check', desc: 'Detections must show consistent apparent motion rate' },
              { step: '2', title: 'Orbital Plane Test', desc: 'All observations must lie on a valid orbital plane' },
              { step: '3', title: 'Time Consistency', desc: 'Spacing matches expected orbital period' },
              { step: '4', title: 'Brightness Correlation', desc: 'Object magnitude remains consistent across sensors' },
            ].map((item, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  gap: 14,
                  padding: 16,
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: 8,
                  border: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <div style={{
                  width: 32,
                  height: 32,
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #00d4ff, #7c3aed)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 14,
                  fontWeight: 700,
                  flexShrink: 0,
                }}>
                  {item.step}
                </div>
                <div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: '#e6edf3' }}>{item.title}</div>
                  <div style={{ fontSize: 14, color: '#8b949e', marginTop: 4 }}>{item.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  // Gauss IOD Section
  const GaussSection = () => (
    <div style={{ padding: '32px 40px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 40 }}>
        {/* Visual representation */}
        <div style={{
          background: 'rgba(0,0,0,0.3)',
          borderRadius: 16,
          padding: 24,
        }}>
          <svg viewBox="0 0 350 300" style={{ width: '100%', height: 280 }}>
            {/* Central body (Earth) */}
            <circle cx="175" cy="200" r="50" fill="#1a4a7a" opacity="0.4" />
            <text x="175" y="205" textAnchor="middle" fill="#3b82f6" fontSize="10">Earth</text>
            
            {/* Orbit arc */}
            <ellipse 
              cx="175" cy="150" rx="140" ry="80" 
              fill="none" 
              stroke="#374151" 
              strokeWidth="2"
              strokeDasharray="8,4"
            />
            
            {/* Three observation points on orbit */}
            {[
              { x: 60, y: 120, t: 't₁', color: '#00d4ff' },
              { x: 175, y: 70, t: 't₂', color: '#8b5cf6' },
              { x: 290, y: 120, t: 't₃', color: '#10b981' },
            ].map((obs, i) => (
              <g key={i}>
                <ObservationDot x={obs.x} y={obs.y} color={obs.color} delay={i * 20} size={10} label={obs.t} />
              </g>
            ))}

            {/* Position vectors from Earth center */}
            <line x1="175" y1="200" x2="60" y2="120" stroke="#00d4ff" strokeWidth="1.5" opacity="0.6" />
            <line x1="175" y1="200" x2="175" y2="70" stroke="#8b5cf6" strokeWidth="1.5" opacity="0.6" />
            <line x1="175" y1="200" x2="290" y2="120" stroke="#10b981" strokeWidth="1.5" opacity="0.6" />

            {/* Vector labels */}
            <text x="105" y="170" fill="#00d4ff" fontSize="12" fontFamily="IBM Plex Mono">r₁</text>
            <text x="185" y="140" fill="#8b5cf6" fontSize="12" fontFamily="IBM Plex Mono">r₂</text>
            <text x="240" y="170" fill="#10b981" fontSize="12" fontFamily="IBM Plex Mono">r₃</text>

            {/* Computed orbit (revealed) */}
            <ellipse 
              cx="175" cy="150" rx="140" ry="80" 
              fill="none" 
              stroke="#f59e0b" 
              strokeWidth="2"
              opacity={0.4 + Math.sin(animationPhase * 0.05) * 0.3}
            />
          </svg>
        </div>

        {/* Method explanation */}
        <div>
          <h4 style={{ color: '#e6edf3', fontSize: 24, marginBottom: 20, fontWeight: 600 }}>
            Gauss Method (1801)
          </h4>
          <p style={{ fontSize: 16, color: '#8b949e', lineHeight: 1.8, marginBottom: 24 }}>
            Carl Friedrich Gauss developed this method to determine the orbit of Ceres from just 
            three observations. It remains the foundation of modern orbit determination.
          </p>

          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: 12,
            padding: 20,
            fontFamily: 'IBM Plex Mono',
            fontSize: 15,
            marginBottom: 24,
          }}>
            <div style={{ color: '#8b949e', marginBottom: 12, fontSize: 14 }}>// Core equations</div>
            <div style={{ color: '#00d4ff', lineHeight: 1.8 }}>
              <div>τ₁ = k(t₁ - t₂)</div>
              <div>τ₃ = k(t₃ - t₂)</div>
              <div style={{ marginTop: 10, color: '#f59e0b' }}>
                r₂ = (τ₃/τ)r₁ - (τ₁τ₃/τ)r₂* + (τ₁/τ)r₃
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { input: '3 position observations', output: '6 orbital elements' },
              { input: 'Time stamps (t₁, t₂, t₃)', output: 'Semi-major axis (a)' },
              { input: 'RA/Dec angles', output: 'Eccentricity (e), Inclination (i)' },
            ].map((item, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 14, fontSize: 14 }}>
                <span style={{ color: '#8b949e', flex: 1 }}>{item.input}</span>
                <span style={{ color: '#374151' }}>→</span>
                <span style={{ color: '#10b981', flex: 1 }}>{item.output}</span>
              </div>
            ))}
          </div>

          {/* Complexity Notes */}
          <div style={{
            marginTop: 24,
            padding: 20,
            background: 'rgba(139,92,246,0.1)',
            borderRadius: 8,
            border: '1px solid rgba(139,92,246,0.3)',
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#8b5cf6', marginBottom: 12, letterSpacing: '1px' }}>
              COMPUTATIONAL COMPLEXITY
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, fontSize: 14 }}>
              <div>
                <span style={{ color: '#8b949e' }}>Time: </span>
                <span style={{ color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>O(n³)</span>
              </div>
              <div>
                <span style={{ color: '#8b949e' }}>Iterations: </span>
                <span style={{ color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>3-8 typical</span>
              </div>
              <div>
                <span style={{ color: '#8b949e' }}>Convergence: </span>
                <span style={{ color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>~1ms/object</span>
              </div>
              <div>
                <span style={{ color: '#8b949e' }}>Accuracy: </span>
                <span style={{ color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>±100m (3 obs)</span>
              </div>
            </div>
            <div style={{ fontSize: 13, color: '#8b949e', marginTop: 14, lineHeight: 1.6 }}>
              Iterative solution using Newton-Raphson. Requires good initial guess from Lambert problem geometry.
              Well-suited for batch processing of multiple debris objects.
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Herrick-Gibbs Section
  const HerrickGibbsSection = () => (
    <div style={{ padding: '32px 40px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 40 }}>
        {/* Method comparison */}
        <div>
          <h4 style={{ color: '#e6edf3', fontSize: 24, marginBottom: 20, fontWeight: 600 }}>
            Herrick-Gibbs Method
          </h4>
          <p style={{ fontSize: 16, color: '#8b949e', lineHeight: 1.8, marginBottom: 24 }}>
            Optimized for closely-spaced observations (small time separations). Uses numerical 
            differentiation to estimate velocity directly from position vectors.
          </p>

          <div style={{
            background: 'rgba(0,0,0,0.3)',
            borderRadius: 12,
            padding: 20,
            fontFamily: 'IBM Plex Mono',
            fontSize: 15,
            marginBottom: 24,
          }}>
            <div style={{ color: '#8b949e', marginBottom: 12, fontSize: 14 }}>// Velocity estimation</div>
            <div style={{ color: '#ec4899', lineHeight: 1.8 }}>
              <div>v₂ = -Δt₃₂/(Δt₂₁·Δt₃₁) · r₁</div>
              <div style={{ marginTop: 6 }}>   + (Δt₃₂-Δt₂₁)/(Δt₂₁·Δt₃₂) · r₂</div>
              <div style={{ marginTop: 6 }}>   + Δt₂₁/(Δt₃₂·Δt₃₁) · r₃</div>
            </div>
          </div>

          {/* When to use which */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 16,
          }}>
            <div style={{
              padding: 20,
              background: 'rgba(139,92,246,0.1)',
              borderRadius: 8,
              border: '1px solid rgba(139,92,246,0.3)',
            }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: '#8b5cf6', marginBottom: 12 }}>
                USE GAUSS
              </div>
              <div style={{ fontSize: 14, color: '#8b949e', lineHeight: 1.7 }}>
                • Large time gaps<br/>
                • &gt; 10° angular separation<br/>
                • Different orbital positions
              </div>
            </div>
            <div style={{
              padding: 20,
              background: 'rgba(236,72,153,0.1)',
              borderRadius: 8,
              border: '1px solid rgba(236,72,153,0.3)',
            }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: '#ec4899', marginBottom: 12 }}>
                USE HERRICK-GIBBS
              </div>
              <div style={{ fontSize: 14, color: '#8b949e', lineHeight: 1.7 }}>
                • Small time gaps<br/>
                • &lt; 5° angular separation<br/>
                • Rapid tracking scenarios
              </div>
            </div>
          </div>
        </div>

        {/* Visual comparison */}
        <div style={{
          background: 'rgba(0,0,0,0.3)',
          borderRadius: 16,
          padding: 24,
        }}>
          <svg viewBox="0 0 300 280" style={{ width: '100%', height: 260 }}>
            {/* Orbit segment */}
            <path
              d="M 30 200 Q 150 40 270 200"
              fill="none"
              stroke="#374151"
              strokeWidth="2"
            />
            
            {/* Closely spaced observations */}
            <ObservationDot x={80} y={140} color="#ec4899" delay={0} size={8} label="r₁" />
            <ObservationDot x={150} y={80} color="#ec4899" delay={10} size={8} label="r₂" />
            <ObservationDot x={220} y={140} color="#ec4899" delay={20} size={8} label="r₃" />

            {/* Velocity vector at r₂ */}
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" />
              </marker>
            </defs>
            <line 
              x1="150" y1="80" 
              x2="200" y2="50" 
              stroke="#f59e0b" 
              strokeWidth="3"
              markerEnd="url(#arrowhead)"
              opacity={0.6 + Math.sin(animationPhase * 0.1) * 0.3}
            />
            <text x="210" y="45" fill="#f59e0b" fontSize="12" fontFamily="IBM Plex Mono">v₂</text>

            {/* Delta t indicators */}
            <line x1="80" y1="170" x2="150" y2="170" stroke="#8b949e" strokeWidth="1" />
            <text x="115" y="185" textAnchor="middle" fill="#8b949e" fontSize="10">Δt₂₁</text>
            
            <line x1="150" y1="170" x2="220" y2="170" stroke="#8b949e" strokeWidth="1" />
            <text x="185" y="185" textAnchor="middle" fill="#8b949e" fontSize="10">Δt₃₂</text>

            {/* Label */}
            <text x="150" y="250" textAnchor="middle" fill="#ec4899" fontSize="12" fontWeight="600">
              Small Δt → Use Herrick-Gibbs
            </text>
          </svg>

          {/* Complexity Notes */}
          <div style={{
            marginTop: 16,
            padding: 20,
            background: 'rgba(236,72,153,0.1)',
            borderRadius: 8,
            border: '1px solid rgba(236,72,153,0.3)',
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#ec4899', marginBottom: 12, letterSpacing: '1px' }}>
              COMPUTATIONAL COMPLEXITY
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, fontSize: 14 }}>
              <div>
                <span style={{ color: '#8b949e' }}>Time: </span>
                <span style={{ color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>O(1)</span>
              </div>
              <div>
                <span style={{ color: '#8b949e' }}>Iterations: </span>
                <span style={{ color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>None (closed-form)</span>
              </div>
              <div>
                <span style={{ color: '#8b949e' }}>Execution: </span>
                <span style={{ color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>&lt;0.1ms/object</span>
              </div>
              <div>
                <span style={{ color: '#8b949e' }}>Accuracy: </span>
                <span style={{ color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>±50m (close obs)</span>
              </div>
            </div>
            <div style={{ fontSize: 13, color: '#8b949e', marginTop: 14, lineHeight: 1.6 }}>
              Direct algebraic solution—no iteration required. Higher accuracy than Gauss when observations 
              are closely spaced (&lt;60 seconds apart). Ideal for real-time tracking scenarios.
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Covariance Section
  const CovarianceSection = () => (
    <div style={{ padding: '32px 40px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 40 }}>
        {/* Covariance ellipse visualization */}
        <div style={{
          background: 'rgba(0,0,0,0.3)',
          borderRadius: 16,
          padding: 24,
        }}>
          <svg viewBox="0 0 350 280" style={{ width: '100%', height: 260 }}>
            {/* Grid */}
            {[0, 1, 2, 3, 4].map(i => (
              <g key={i}>
                <line x1="50" y1={50 + i * 45} x2="300" y2={50 + i * 45} stroke="#374151" strokeWidth="0.5" opacity="0.3" />
                <line x1={50 + i * 62.5} y1="50" x2={50 + i * 62.5} y2="230" stroke="#374151" strokeWidth="0.5" opacity="0.3" />
              </g>
            ))}
            
            {/* Labels */}
            <text x="175" y="265" textAnchor="middle" fill="#8b949e" fontSize="11">Cross-track (km)</text>
            <text x="20" y="140" textAnchor="middle" fill="#8b949e" fontSize="11" transform="rotate(-90, 20, 140)">Along-track (km)</text>

            {/* Large uncertainty ellipse (1 obs) */}
            <ellipse
              cx="175" cy="140"
              rx={100 - animationPhase * 0.3}
              ry={60 - animationPhase * 0.2}
              fill="rgba(239,68,68,0.1)"
              stroke="#ef4444"
              strokeWidth="2"
              opacity={Math.max(0.1, 1 - animationPhase * 0.01)}
            />

            {/* Medium uncertainty ellipse (2 obs) */}
            <ellipse
              cx="175" cy="140"
              rx={Math.max(20, 60 - animationPhase * 0.4)}
              ry={Math.max(15, 35 - animationPhase * 0.25)}
              fill="rgba(245,158,11,0.1)"
              stroke="#f59e0b"
              strokeWidth="2"
              opacity={animationPhase > 30 ? Math.min(1, (animationPhase - 30) * 0.03) : 0}
            />

            {/* Small uncertainty ellipse (3+ obs) */}
            <ellipse
              cx="175" cy="140"
              rx={20}
              ry={12}
              fill="rgba(16,185,129,0.2)"
              stroke="#10b981"
              strokeWidth="2"
              opacity={animationPhase > 60 ? Math.min(1, (animationPhase - 60) * 0.03) : 0}
            />

            {/* Center point (true position) */}
            <circle cx="175" cy="140" r="4" fill="#fff" />

            {/* Legend */}
            <g transform="translate(50, 50)">
              <rect x="0" y="0" width="12" height="12" fill="rgba(239,68,68,0.3)" stroke="#ef4444" />
              <text x="18" y="10" fill="#8b949e" fontSize="10">1 observation</text>
              
              <rect x="0" y="20" width="12" height="12" fill="rgba(245,158,11,0.3)" stroke="#f59e0b" />
              <text x="18" y="30" fill="#8b949e" fontSize="10">2 observations</text>
              
              <rect x="0" y="40" width="12" height="12" fill="rgba(16,185,129,0.3)" stroke="#10b981" />
              <text x="18" y="50" fill="#8b949e" fontSize="10">3+ observations</text>
            </g>
          </svg>
        </div>

        {/* Explanation */}
        <div>
          <h4 style={{ color: '#e6edf3', fontSize: 24, marginBottom: 20, fontWeight: 600 }}>
            Covariance Estimation
          </h4>
          <p style={{ fontSize: 16, color: '#8b949e', lineHeight: 1.8, marginBottom: 24 }}>
            The covariance matrix quantifies our uncertainty about the object's position and velocity. 
            More observations from different geometries shrink this uncertainty ellipsoid.
          </p>

          {/* Metrics */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {[
              { obs: '1 obs', sigma: '± 50 km', color: '#ef4444', confidence: '~30%' },
              { obs: '2 obs', sigma: '± 5 km', color: '#f59e0b', confidence: '~70%' },
              { obs: '3 obs', sigma: '± 500 m', color: '#10b981', confidence: '~95%' },
              { obs: '5+ obs', sigma: '± 50 m', color: '#00d4ff', confidence: '~99%' },
            ].map((item, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: 14,
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: 8,
                  border: `1px solid ${item.color}33`,
                }}
              >
                <div style={{ 
                  width: 80, 
                  fontSize: 14, 
                  color: item.color,
                  fontWeight: 600,
                }}>
                  {item.obs}
                </div>
                <div style={{ 
                  flex: 1, 
                  fontSize: 16, 
                  color: '#e6edf3',
                  fontFamily: 'IBM Plex Mono',
                }}>
                  {item.sigma}
                </div>
                <div style={{ 
                  fontSize: 13, 
                  color: '#8b949e',
                  background: 'rgba(255,255,255,0.05)',
                  padding: '6px 10px',
                  borderRadius: 4,
                }}>
                  {item.confidence}
                </div>
              </div>
            ))}
          </div>

          <div style={{
            marginTop: 24,
            padding: 20,
            background: 'rgba(0,212,255,0.05)',
            borderRadius: 8,
            border: '1px solid rgba(0,212,255,0.2)',
          }}>
            <div style={{ fontSize: 14, color: '#00d4ff', fontWeight: 600, marginBottom: 8 }}>
              CONJUNCTION ASSESSMENT IMPACT
            </div>
            <p style={{ fontSize: 14, color: '#8b949e', margin: 0, lineHeight: 1.7 }}>
              Smaller covariance → more accurate collision probability (Pc) calculation → 
              better GO/NO-GO decisions with fewer false alarms.
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  // Industry Standards Section
  const IndustryStandardsSection = () => (
    <div style={{ padding: '32px 40px' }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <p style={{ fontSize: 16, color: '#8b949e', lineHeight: 1.8, maxWidth: 900 }}>
          AVERA-ATLAS implements the same mathematical foundations used by government and commercial 
          space surveillance organizations worldwide. Here's how our approach compares to industry standards.
        </p>
      </div>

      {/* Main comparison grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        
        {/* Government Standards */}
        <div style={{
          background: 'rgba(0,0,0,0.3)',
          borderRadius: 16,
          padding: 28,
          border: '1px solid rgba(59,130,246,0.3)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
            <div style={{
              width: 48,
              height: 48,
              borderRadius: 8,
              background: 'linear-gradient(135deg, #1e40af, #3b82f6)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 22,
            }}>
              🛡️
            </div>
            <div>
              <div style={{ fontSize: 17, fontWeight: 600, color: '#e6edf3' }}>18th Space Defense Squadron</div>
              <div style={{ fontSize: 13, color: '#8b949e' }}>U.S. Space Force • Vandenberg SFB</div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {[
              { label: 'Catalog Size', value: '47,000+ objects', note: 'Tracked in Space Catalog' },
              { label: 'Sensor Network', value: 'Space Fence + Ground', note: 'S-band radar array + optical' },
              { label: 'IOD Methods', value: 'Gauss, Double-r, Gooding', note: 'Multiple algorithms' },
              { label: 'Update Rate', value: '~Daily per object', note: 'Routine observations' },
              { label: 'CDMs Issued', value: '~1M/year', note: 'Conjunction Data Messages' },
            ].map((item, i) => (
              <div key={i} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px 16px',
                background: 'rgba(59,130,246,0.05)',
                borderRadius: 6,
              }}>
                <div>
                  <div style={{ fontSize: 14, color: '#e6edf3' }}>{item.label}</div>
                  <div style={{ fontSize: 12, color: '#8b949e' }}>{item.note}</div>
                </div>
                <div style={{ fontSize: 14, color: '#3b82f6', fontFamily: 'IBM Plex Mono', fontWeight: 500 }}>
                  {item.value}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Commercial Providers */}
        <div style={{
          background: 'rgba(0,0,0,0.3)',
          borderRadius: 16,
          padding: 28,
          border: '1px solid rgba(168,85,247,0.3)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
            <div style={{
              width: 48,
              height: 48,
              borderRadius: 8,
              background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 22,
            }}>
              📡
            </div>
            <div>
              <div style={{ fontSize: 17, fontWeight: 600, color: '#e6edf3' }}>Commercial SSA Providers</div>
              <div style={{ fontSize: 13, color: '#8b949e' }}>LeoLabs • ExoAnalytic • Numerica</div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {[
              { label: 'LeoLabs', value: 'Phased-array radar', note: '6 ground stations worldwide' },
              { label: 'ExoAnalytic', value: 'Optical network', note: '300+ telescopes globally' },
              { label: 'Numerica', value: 'Hybrid radar/optical', note: 'ML-enhanced tracking' },
              { label: 'Update Rate', value: 'Minutes to hours', note: 'Higher-cadence than 18th SDS' },
              { label: 'Focus', value: 'LEO/MEO commercial', note: 'Constellation operators' },
            ].map((item, i) => (
              <div key={i} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px 16px',
                background: 'rgba(168,85,247,0.05)',
                borderRadius: 6,
              }}>
                <div>
                  <div style={{ fontSize: 14, color: '#e6edf3' }}>{item.label}</div>
                  <div style={{ fontSize: 12, color: '#8b949e' }}>{item.note}</div>
                </div>
                <div style={{ fontSize: 14, color: '#a855f7', fontFamily: 'IBM Plex Mono', fontWeight: 500 }}>
                  {item.value}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* AVERA-ATLAS Positioning */}
      <div style={{
        marginTop: 24,
        background: 'linear-gradient(135deg, rgba(0,212,255,0.1), rgba(16,185,129,0.1))',
        borderRadius: 16,
        padding: 24,
        border: '1px solid rgba(0,212,255,0.3)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
          <div style={{
            width: 40,
            height: 40,
            borderRadius: 8,
            background: 'linear-gradient(135deg, #00d4ff, #10b981)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 18,
          }}>
            ⚡
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#e6edf3' }}>AVERA-ATLAS Approach</div>
            <div style={{ fontSize: 11, color: '#8b949e' }}>Distributed CubeSat constellation • Real-time processing</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {[
            { label: 'Sensor Type', value: 'SWIR CubeSats', icon: '🛰️' },
            { label: 'IOD Methods', value: 'Gauss + H-G', icon: '📐' },
            { label: 'Processing', value: 'Edge + Cloud', icon: '☁️' },
            { label: 'Differentiator', value: 'Multi-sensor fusion', icon: '🎯' },
          ].map((item, i) => (
            <div key={i} style={{
              textAlign: 'center',
              padding: 16,
              background: 'rgba(0,0,0,0.3)',
              borderRadius: 8,
            }}>
              <div style={{ fontSize: 24, marginBottom: 8 }}>{item.icon}</div>
              <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 4 }}>{item.label}</div>
              <div style={{ fontSize: 12, color: '#00d4ff', fontWeight: 600 }}>{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm Comparison Table */}
      <div style={{
        marginTop: 24,
        background: 'rgba(0,0,0,0.3)',
        borderRadius: 16,
        padding: 24,
      }}>
        <h4 style={{ fontSize: 13, color: '#8b949e', marginBottom: 16, letterSpacing: '1px' }}>
          ALGORITHM COMPARISON
        </h4>
        
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <th style={{ textAlign: 'left', padding: '12px 16px', color: '#8b949e', fontWeight: 500 }}>Method</th>
                <th style={{ textAlign: 'left', padding: '12px 16px', color: '#8b949e', fontWeight: 500 }}>Used By</th>
                <th style={{ textAlign: 'left', padding: '12px 16px', color: '#8b949e', fontWeight: 500 }}>Best For</th>
                <th style={{ textAlign: 'center', padding: '12px 16px', color: '#8b949e', fontWeight: 500 }}>AVERA-ATLAS</th>
              </tr>
            </thead>
            <tbody>
              {[
                { method: 'Gauss', usedBy: '18th SDS, NASA, ESA', bestFor: 'Sparse observations (>10° sep)', supported: true },
                { method: 'Herrick-Gibbs', usedBy: '18th SDS, AGI STK', bestFor: 'Rapid tracking (<5° sep)', supported: true },
                { method: 'Double-r', usedBy: '18th SDS', bestFor: 'Optical angles-only', supported: false, planned: true },
                { method: 'Gooding', usedBy: 'ESA, Commercial', bestFor: 'Lambert-based refinement', supported: false, planned: true },
                { method: 'Kalman Filter', usedBy: 'All major providers', bestFor: 'State estimation + updates', supported: true },
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '12px 16px', color: '#e6edf3', fontFamily: 'IBM Plex Mono' }}>{row.method}</td>
                  <td style={{ padding: '12px 16px', color: '#8b949e' }}>{row.usedBy}</td>
                  <td style={{ padding: '12px 16px', color: '#8b949e' }}>{row.bestFor}</td>
                  <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                    {row.supported ? (
                      <span style={{ color: '#10b981', fontWeight: 600 }}>✓ Implemented</span>
                    ) : row.planned ? (
                      <span style={{ color: '#f59e0b' }}>◐ Roadmap</span>
                    ) : (
                      <span style={{ color: '#6b7280' }}>—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Standards Compliance */}
      <div style={{
        marginTop: 24,
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 16,
      }}>
        {[
          { 
            standard: 'CCSDS 508.0', 
            title: 'Conjunction Data Message', 
            status: 'Compliant',
            color: '#10b981',
            desc: 'Standard CDM format for conjunction warnings'
          },
          { 
            standard: 'NASA-STD-3001', 
            title: 'Collision Probability', 
            status: 'Aligned',
            color: '#10b981',
            desc: 'Pc calculation per NASA methodology'
          },
          { 
            standard: 'ISO 27875', 
            title: 'Space Debris Mitigation', 
            status: 'Supporting',
            color: '#f59e0b',
            desc: 'Debris tracking for mitigation decisions'
          },
        ].map((item, i) => (
          <div key={i} style={{
            padding: 16,
            background: 'rgba(0,0,0,0.3)',
            borderRadius: 12,
            border: `1px solid ${item.color}33`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: 11, color: '#8b949e', fontFamily: 'IBM Plex Mono' }}>{item.standard}</span>
              <span style={{ fontSize: 10, color: item.color, fontWeight: 600 }}>{item.status}</span>
            </div>
            <div style={{ fontSize: 13, color: '#e6edf3', marginBottom: 6, fontWeight: 500 }}>{item.title}</div>
            <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.5 }}>{item.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderSection = () => {
    switch (sections[activeSection].id) {
      case 'problem': return <ProblemSection />;
      case 'correlation': return <CorrelationSection />;
      case 'gauss': return <GaussSection />;
      case 'herrick': return <HerrickGibbsSection />;
      case 'covariance': return <CovarianceSection />;
      case 'industry': return <IndustryStandardsSection />;
      default: return <ProblemSection />;
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Section header */}
      <div style={{
        padding: '20px 32px',
        borderBottom: '1px solid rgba(255,255,255,0.1)',
        background: 'rgba(0,0,0,0.2)',
      }}>
        <h2 style={{ 
          fontSize: 20, 
          fontWeight: 300, 
          color: '#8b949e', 
          letterSpacing: '3px',
          marginBottom: 16,
        }}>
          TRACKER SERVICE — ALGORITHM DEEP DIVE
        </h2>
        
        {/* Section tabs */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {sections.map((section, index) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(index)}
              style={{
                padding: '12px 20px',
                background: activeSection === index 
                  ? 'linear-gradient(135deg, rgba(236,72,153,0.2), rgba(139,92,246,0.2))'
                  : 'rgba(255,255,255,0.03)',
                border: `1px solid ${activeSection === index ? '#ec4899' : 'rgba(255,255,255,0.1)'}`,
                borderRadius: 8,
                color: activeSection === index ? '#ec4899' : '#8b949e',
                fontSize: 14,
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.3s',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
              }}
            >
              <span style={{ 
                fontFamily: 'IBM Plex Mono', 
                fontSize: 16,
                opacity: 0.7,
              }}>
                {section.icon}
              </span>
              {section.title}
            </button>
          ))}
        </div>
      </div>

      {/* Section content */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {renderSection()}
      </div>

      {/* Bottom summary bar */}
      <div style={{
        padding: '16px 32px',
        borderTop: '1px solid rgba(255,255,255,0.1)',
        background: 'rgba(0,0,0,0.3)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div style={{ display: 'flex', gap: 24 }}>
          <div>
            <div style={{ fontSize: 10, color: '#8b949e', letterSpacing: '1px' }}>METHODS</div>
            <div style={{ fontSize: 14, color: '#e6edf3', fontWeight: 500 }}>Gauss + Herrick-Gibbs</div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: '#8b949e', letterSpacing: '1px' }}>MIN OBSERVATIONS</div>
            <div style={{ fontSize: 14, color: '#e6edf3', fontWeight: 500 }}>3 (from different sensors)</div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: '#8b949e', letterSpacing: '1px' }}>OUTPUT</div>
            <div style={{ fontSize: 14, color: '#e6edf3', fontWeight: 500 }}>6 Orbital Elements + Covariance</div>
          </div>
        </div>
        <div style={{
          padding: '8px 16px',
          background: 'rgba(16,185,129,0.1)',
          border: '1px solid rgba(16,185,129,0.3)',
          borderRadius: 8,
          fontSize: 12,
          color: '#10b981',
          fontWeight: 600,
        }}>
          TRACKER SERVICE: OPERATIONAL
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// 3D Orbital Visualization Component - Multi-Sensor Tracking Concept
// Shows 3 CubeSats acquiring observations on the same debris for IOD
// ============================================================================

function OrbitalVisualization({ activeScenario, isPlaying }) {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const objectsRef = useRef({});
  const animationRef = useRef(null);
  const timeRef = useRef(0);
  const initializedRef = useRef(false);

  // CubeSat colors for identification
  const CUBESAT_COLORS = [
    { main: 0x00d4ff, emissive: 0x006680, name: 'ATLAS-1' },  // Cyan
    { main: 0x8b5cf6, emissive: 0x4c2889, name: 'ATLAS-2' },  // Purple
    { main: 0x10b981, emissive: 0x065f46, name: 'ATLAS-3' },  // Green
  ];

  useEffect(() => {
    if (!containerRef.current || initializedRef.current) return;
    initializedRef.current = true;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 20, 35);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(50, 50, 50);
    scene.add(directionalLight);

    // Earth
    const earthGeometry = new THREE.SphereGeometry(5, 32, 32);
    const earthMaterial = new THREE.MeshPhongMaterial({
      color: 0x1a4a7a,
      emissive: 0x0a2040,
      shininess: 25,
    });
    const earth = new THREE.Mesh(earthGeometry, earthMaterial);
    scene.add(earth);
    objectsRef.current.earth = earth;

    // Earth glow
    const glowGeometry = new THREE.SphereGeometry(5.3, 32, 32);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: 0x3b82f6,
      transparent: true,
      opacity: 0.12,
    });
    scene.add(new THREE.Mesh(glowGeometry, glowMaterial));

    // Helper to create orbital path points with inclination
    const createOrbitPoints = (radius, inclination, raan, segments = 128) => {
      const points = [];
      for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        
        // Apply inclination rotation (around X axis)
        const y1 = 0;
        const z1 = z;
        const yInclined = y1 * Math.cos(inclination) - z1 * Math.sin(inclination);
        const zInclined = y1 * Math.sin(inclination) + z1 * Math.cos(inclination);
        
        // Apply RAAN rotation (around Y axis)
        const xFinal = x * Math.cos(raan) + zInclined * Math.sin(raan);
        const zFinal = -x * Math.sin(raan) + zInclined * Math.cos(raan);
        
        points.push(new THREE.Vector3(xFinal, yInclined, zFinal));
      }
      return points;
    };

    // Create 3 CubeSats in different orbital planes
    const cubeSats = [];
    const cubeSatOrbits = [
      { radius: 9, inclination: Math.PI * 0.15, raan: 0, phase: 0 },           // ATLAS-1
      { radius: 9.5, inclination: Math.PI * 0.25, raan: Math.PI * 0.67, phase: Math.PI * 0.4 },  // ATLAS-2
      { radius: 8.5, inclination: Math.PI * 0.1, raan: Math.PI * 1.33, phase: Math.PI * 0.8 },  // ATLAS-3
    ];

    cubeSatOrbits.forEach((orbit, index) => {
      const color = CUBESAT_COLORS[index];
      
      // CubeSat body (box shape)
      const satGeometry = new THREE.BoxGeometry(0.4, 0.4, 0.6);
      const satMaterial = new THREE.MeshPhongMaterial({
        color: color.main,
        emissive: color.emissive,
        shininess: 60,
      });
      const cubeSat = new THREE.Mesh(satGeometry, satMaterial);
      
      // Solar panels
      const panelGeometry = new THREE.BoxGeometry(1.2, 0.05, 0.4);
      const panelMaterial = new THREE.MeshPhongMaterial({
        color: 0x1e3a5f,
        emissive: 0x0a1520,
        shininess: 80,
      });
      const leftPanel = new THREE.Mesh(panelGeometry, panelMaterial);
      leftPanel.position.set(-0.8, 0, 0);
      cubeSat.add(leftPanel);
      
      const rightPanel = new THREE.Mesh(panelGeometry, panelMaterial);
      rightPanel.position.set(0.8, 0, 0);
      cubeSat.add(rightPanel);

      // Store orbital parameters
      cubeSat.userData = { ...orbit, colorIndex: index };
      
      scene.add(cubeSat);
      cubeSats.push(cubeSat);

      // Orbital path
      const orbitPoints = createOrbitPoints(orbit.radius, orbit.inclination, orbit.raan);
      const orbitGeometry = new THREE.BufferGeometry().setFromPoints(orbitPoints);
      const orbitMaterial = new THREE.LineBasicMaterial({
        color: color.main,
        transparent: true,
        opacity: 0.3,
      });
      scene.add(new THREE.Line(orbitGeometry, orbitMaterial));
    });
    objectsRef.current.cubeSats = cubeSats;

    // Target debris object (the one being tracked by all 3 CubeSats)
    const debrisGeometry = new THREE.IcosahedronGeometry(0.35, 0);
    const debrisMaterial = new THREE.MeshPhongMaterial({
      color: 0xef4444,
      emissive: 0x7f1d1d,
      shininess: 40,
    });
    const targetDebris = new THREE.Mesh(debrisGeometry, debrisMaterial);
    targetDebris.userData = {
      radius: 10.5,
      inclination: Math.PI * 0.18,
      raan: Math.PI * 0.3,
      phase: 0,
      period: 1.2,
    };
    scene.add(targetDebris);
    objectsRef.current.targetDebris = targetDebris;

    // Target debris glow
    const debrisGlowGeometry = new THREE.IcosahedronGeometry(0.5, 0);
    const debrisGlowMaterial = new THREE.MeshBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.2,
    });
    const debrisGlow = new THREE.Mesh(debrisGlowGeometry, debrisGlowMaterial);
    scene.add(debrisGlow);
    objectsRef.current.debrisGlow = debrisGlow;

    // Debris orbit path
    const debrisOrbitPoints = createOrbitPoints(
      targetDebris.userData.radius,
      targetDebris.userData.inclination,
      targetDebris.userData.raan
    );
    const debrisOrbitGeometry = new THREE.BufferGeometry().setFromPoints(debrisOrbitPoints);
    const debrisOrbitMaterial = new THREE.LineBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.25,
    });
    scene.add(new THREE.Line(debrisOrbitGeometry, debrisOrbitMaterial));

    // Observation lines (from each CubeSat to debris)
    const observationLines = [];
    CUBESAT_COLORS.forEach((color, index) => {
      const linePositions = new Float32Array([0, 0, 0, 1, 0, 0]);
      const lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
      const lineMaterial = new THREE.LineBasicMaterial({
        color: color.main,
        transparent: true,
        opacity: 0.6,
      });
      const line = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(line);
      observationLines.push(line);
    });
    objectsRef.current.observationLines = observationLines;

    // Observation pulse indicators (spheres that travel along the observation lines)
    const observationPulses = [];
    CUBESAT_COLORS.forEach((color) => {
      const pulseGeometry = new THREE.SphereGeometry(0.1, 8, 8);
      const pulseMaterial = new THREE.MeshBasicMaterial({
        color: color.main,
        transparent: true,
        opacity: 0.9,
      });
      const pulse = new THREE.Mesh(pulseGeometry, pulseMaterial);
      pulse.visible = false;
      scene.add(pulse);
      observationPulses.push(pulse);
    });
    objectsRef.current.observationPulses = observationPulses;

    // Additional background debris (not being tracked - for visual context)
    const backgroundDebris = [];
    for (let i = 0; i < 8; i++) {
      const bgDebrisGeometry = new THREE.OctahedronGeometry(0.12, 0);
      const bgDebrisMaterial = new THREE.MeshPhongMaterial({
        color: 0x6b7280,
        emissive: 0x374151,
        shininess: 20,
      });
      const bgDebris = new THREE.Mesh(bgDebrisGeometry, bgDebrisMaterial);
      bgDebris.userData = {
        radius: 7 + Math.random() * 6,
        inclination: (Math.random() - 0.5) * Math.PI * 0.4,
        raan: Math.random() * Math.PI * 2,
        phase: Math.random() * Math.PI * 2,
        period: 0.8 + Math.random() * 1.5,
      };
      scene.add(bgDebris);
      backgroundDebris.push(bgDebris);
    }
    objectsRef.current.backgroundDebris = backgroundDebris;

    // Stars background
    const starPositions = [];
    for (let i = 0; i < 1500; i++) {
      const x = (Math.random() - 0.5) * 400;
      const y = (Math.random() - 0.5) * 400;
      const z = (Math.random() - 0.5) * 400;
      if (Math.sqrt(x*x + y*y + z*z) > 50) {
        starPositions.push(x, y, z);
      }
    }
    const starsGeometry = new THREE.BufferGeometry();
    starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
    const starsMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.5,
      transparent: true,
      opacity: 0.8,
    });
    scene.add(new THREE.Points(starsGeometry, starsMaterial));

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      initializedRef.current = false;
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (rendererRef.current) rendererRef.current.dispose();
    };
  }, []);

  // Animation loop
  useEffect(() => {
    if (!sceneRef.current || !cameraRef.current || !rendererRef.current) return;

    let lastTime = performance.now();

    const getOrbitalPosition = (params, time) => {
      const angle = (time / (params.period || 1)) + (params.phase || 0);
      const x = Math.cos(angle) * params.radius;
      const z = Math.sin(angle) * params.radius;
      
      // Apply inclination
      const yInclined = -z * Math.sin(params.inclination);
      const zInclined = z * Math.cos(params.inclination);
      
      // Apply RAAN
      const xFinal = x * Math.cos(params.raan) + zInclined * Math.sin(params.raan);
      const zFinal = -x * Math.sin(params.raan) + zInclined * Math.cos(params.raan);
      
      return new THREE.Vector3(xFinal, yInclined, zFinal);
    };

    const animate = (currentTime) => {
      const deltaTime = (currentTime - lastTime) / 1000;
      lastTime = currentTime;

      if (isPlaying) {
        timeRef.current += deltaTime;
      }

      const time = timeRef.current;

      // Rotate Earth
      if (objectsRef.current.earth) {
        objectsRef.current.earth.rotation.y = time * 0.05;
      }

      // Update CubeSat positions
      if (objectsRef.current.cubeSats) {
        objectsRef.current.cubeSats.forEach((cubeSat, index) => {
          const pos = getOrbitalPosition(cubeSat.userData, time * 0.4);
          cubeSat.position.copy(pos);
          
          // Orient CubeSat to face direction of travel
          const futurePos = getOrbitalPosition(cubeSat.userData, time * 0.4 + 0.01);
          cubeSat.lookAt(futurePos);
        });
      }

      // Update target debris position
      if (objectsRef.current.targetDebris) {
        const debrisPos = getOrbitalPosition(objectsRef.current.targetDebris.userData, time * 0.3);
        objectsRef.current.targetDebris.position.copy(debrisPos);
        objectsRef.current.targetDebris.rotation.x = time * 0.5;
        objectsRef.current.targetDebris.rotation.y = time * 0.3;
        
        if (objectsRef.current.debrisGlow) {
          objectsRef.current.debrisGlow.position.copy(debrisPos);
          // Pulsing glow
          const pulseScale = 1 + Math.sin(time * 3) * 0.2;
          objectsRef.current.debrisGlow.scale.setScalar(pulseScale);
        }
      }

      // Update observation lines and pulses
      if (objectsRef.current.observationLines && objectsRef.current.cubeSats && objectsRef.current.targetDebris) {
        const debrisPos = objectsRef.current.targetDebris.position;
        
        objectsRef.current.cubeSats.forEach((cubeSat, index) => {
          const line = objectsRef.current.observationLines[index];
          const pulse = objectsRef.current.observationPulses[index];
          
          // Check if CubeSat can "see" the debris (simple visibility check)
          const distance = cubeSat.position.distanceTo(debrisPos);
          const canObserve = distance < 15; // Within observation range
          
          if (canObserve) {
            // Update line positions
            const posAttr = line.geometry.getAttribute('position');
            posAttr.array[0] = cubeSat.position.x;
            posAttr.array[1] = cubeSat.position.y;
            posAttr.array[2] = cubeSat.position.z;
            posAttr.array[3] = debrisPos.x;
            posAttr.array[4] = debrisPos.y;
            posAttr.array[5] = debrisPos.z;
            posAttr.needsUpdate = true;
            
            line.material.opacity = 0.6;
            
            // Animate pulse along the line
            pulse.visible = true;
            const pulseT = ((time * 2 + index * 0.3) % 1);
            pulse.position.lerpVectors(cubeSat.position, debrisPos, pulseT);
            pulse.scale.setScalar(1 + Math.sin(pulseT * Math.PI) * 0.5);
          } else {
            line.material.opacity = 0.1;
            pulse.visible = false;
          }
        });
      }

      // Update background debris
      if (objectsRef.current.backgroundDebris) {
        objectsRef.current.backgroundDebris.forEach((debris) => {
          const pos = getOrbitalPosition(debris.userData, time * 0.25);
          debris.position.copy(pos);
          debris.rotation.x = time * 0.3;
          debris.rotation.z = time * 0.2;
        });
      }

      // Rotate camera
      if (cameraRef.current) {
        const cameraAngle = time * 0.08;
        const cameraRadius = 38;
        cameraRef.current.position.x = Math.sin(cameraAngle) * cameraRadius;
        cameraRef.current.position.z = Math.cos(cameraAngle) * cameraRadius;
        cameraRef.current.position.y = 18 + Math.sin(time * 0.15) * 5;
        cameraRef.current.lookAt(0, 0, 0);
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isPlaying]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div 
        ref={containerRef} 
        style={{ 
          width: '100%', 
          height: '100%',
          background: 'radial-gradient(ellipse at center, #0a1628 0%, #000000 100%)',
        }} 
      />
      {/* Overlay legend */}
      <div style={{
        position: 'absolute',
        top: 20,
        left: 20,
        background: 'rgba(0,0,0,0.7)',
        padding: '16px 20px',
        borderRadius: 12,
        border: '1px solid rgba(255,255,255,0.1)',
      }}>
        <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 12, letterSpacing: '2px', fontWeight: 600 }}>
          MULTI-SENSOR ACQUISITION
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {CUBESAT_COLORS.map((color, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{
                width: 12,
                height: 12,
                borderRadius: 3,
                background: `#${color.main.toString(16).padStart(6, '0')}`,
                boxShadow: `0 0 8px #${color.main.toString(16).padStart(6, '0')}66`,
              }} />
              <span style={{ fontSize: 12, color: '#e6edf3' }}>{color.name}</span>
            </div>
          ))}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 4 }}>
            <div style={{
              width: 12,
              height: 12,
              borderRadius: 3,
              background: '#ef4444',
              boxShadow: '0 0 8px #ef444466',
            }} />
            <span style={{ fontSize: 12, color: '#e6edf3' }}>Target Debris</span>
          </div>
        </div>
      </div>
      {/* IOD Status */}
      <div style={{
        position: 'absolute',
        bottom: 20,
        left: 20,
        background: 'rgba(0,0,0,0.7)',
        padding: '16px 20px',
        borderRadius: 12,
        border: '1px solid rgba(16,185,129,0.3)',
      }}>
        <div style={{ fontSize: 11, color: '#10b981', marginBottom: 8, letterSpacing: '2px', fontWeight: 600 }}>
          IOD STATUS
        </div>
        <div style={{ fontSize: 13, color: '#e6edf3' }}>
          3 sensors tracking • Gauss IOD ready
        </div>
        <div style={{ fontSize: 11, color: '#8b949e', marginTop: 4 }}>
          Cross-correlating observations...
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Animated Pipeline Component
// ============================================================================

function PipelineStage({ stage, index, isActive, isComplete, delay }) {
  const icons = {
    ingest: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: 28, height: 28 }}>
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
      </svg>
    ),
    detect: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: 28, height: 28 }}>
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
      </svg>
    ),
    track: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: 28, height: 28 }}>
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
    propagate: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: 28, height: 28 }}>
        <circle cx="12" cy="12" r="10" />
        <ellipse cx="12" cy="12" rx="10" ry="4" />
        <path d="M12 2v20" />
      </svg>
    ),
    assess: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: 28, height: 28 }}>
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01" />
      </svg>
    ),
    decide: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: 28, height: 28 }}>
        <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
        <path d="M22 4L12 14.01l-3-3" />
      </svg>
    ),
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 12,
        opacity: isActive || isComplete ? 1 : 0.4,
        transform: `scale(${isActive ? 1.05 : 1})`,
        transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        transitionDelay: `${delay}ms`,
      }}
    >
      <div
        style={{
          width: 72,
          height: 72,
          borderRadius: 16,
          background: isActive 
            ? `linear-gradient(135deg, ${stage.color}22, ${stage.color}44)`
            : isComplete 
              ? `linear-gradient(135deg, ${stage.color}11, ${stage.color}22)`
              : 'rgba(255,255,255,0.03)',
          border: `2px solid ${isActive ? stage.color : isComplete ? stage.color + '66' : 'rgba(255,255,255,0.1)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: isActive || isComplete ? stage.color : '#8b949e',
          boxShadow: isActive ? `0 0 30px ${stage.color}44, 0 0 60px ${stage.color}22` : 'none',
          transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
      >
        {icons[stage.icon]}
      </div>
      <div style={{ textAlign: 'center' }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: isActive || isComplete ? '#e6edf3' : '#8b949e',
            letterSpacing: '0.5px',
            transition: 'color 0.3s',
          }}
        >
          {stage.name}
        </div>
        <div
          style={{
            fontSize: 11,
            color: '#8b949e',
            marginTop: 4,
            maxWidth: 100,
          }}
        >
          {stage.service}
        </div>
      </div>
    </div>
  );
}

function DataFlowArrow({ isActive, delay }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 60,
        height: 72,
        position: 'relative',
      }}
    >
      <div
        style={{
          width: 40,
          height: 2,
          background: isActive 
            ? 'linear-gradient(90deg, #00d4ff, #7c3aed)'
            : 'rgba(255,255,255,0.1)',
          position: 'relative',
          transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
          transitionDelay: `${delay}ms`,
        }}
      >
        {isActive && (
          <div
            style={{
              position: 'absolute',
              top: -3,
              left: 0,
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: '#00d4ff',
              boxShadow: '0 0 10px #00d4ff',
              animation: 'flowPulse 1s ease-in-out infinite',
            }}
          />
        )}
        <div
          style={{
            position: 'absolute',
            right: -6,
            top: -4,
            width: 0,
            height: 0,
            borderTop: '5px solid transparent',
            borderBottom: '5px solid transparent',
            borderLeft: isActive ? '8px solid #7c3aed' : '8px solid rgba(255,255,255,0.1)',
            transition: 'border-left-color 0.3s',
          }}
        />
      </div>
    </div>
  );
}

function AnimatedPipeline({ currentStage }) {
  const stages = [
    { id: 'ingest', name: 'INGEST', service: 'Data Acquisition', icon: 'ingest', color: '#00d4ff' },
    { id: 'detect', name: 'DETECT', service: 'YOLOv8 Detection', icon: 'detect', color: '#8b5cf6' },
    { id: 'track', name: 'TRACK', service: 'Multi-Sensor IOD', icon: 'track', color: '#ec4899' },
    { id: 'propagate', name: 'PROPAGATE', service: 'Keplerian Dynamics', icon: 'propagate', color: '#f59e0b' },
    { id: 'assess', name: 'ASSESS', service: 'Pc Calculation', icon: 'assess', color: '#ef4444' },
    { id: 'decide', name: 'DECIDE', service: 'GO/NO-GO', icon: 'decide', color: '#10b981' },
  ];

  const currentIndex = stages.findIndex(s => s.id === currentStage);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        padding: '32px 16px',
        flexWrap: 'wrap',
        gap: '8px',
      }}
    >
      <style>
        {`
          @keyframes flowPulse {
            0%, 100% { transform: translateX(0); opacity: 1; }
            50% { transform: translateX(32px); opacity: 0.5; }
          }
        `}
      </style>
      {stages.map((stage, index) => (
        <React.Fragment key={stage.id}>
          <PipelineStage
            stage={stage}
            index={index}
            isActive={index === currentIndex}
            isComplete={index < currentIndex}
            delay={index * 100}
          />
          {index < stages.length - 1 && (
            <DataFlowArrow 
              isActive={index < currentIndex}
              delay={index * 100 + 50}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

// ============================================================================
// Risk Classification Display
// ============================================================================

function RiskDisplay({ riskLevel, probability, missDistance }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 24,
        padding: 20,
        background: 'rgba(0,0,0,0.4)',
        borderRadius: 12,
        border: '1px solid rgba(255,255,255,0.1)',
      }}
    >
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 6, letterSpacing: '1px' }}>
          RISK LEVEL
        </div>
        <div
          style={{
            fontSize: 28,
            fontWeight: 700,
            color: RISK_COLORS[riskLevel],
            textShadow: `0 0 20px ${RISK_COLORS[riskLevel]}66`,
          }}
        >
          {riskLevel}
        </div>
      </div>
      <div style={{ width: 1, background: 'rgba(255,255,255,0.1)' }} />
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 6, letterSpacing: '1px' }}>
          COLLISION PROB
        </div>
        <div style={{ fontSize: 24, fontWeight: 600, color: '#e6edf3', fontFamily: 'monospace' }}>
          {probability}
        </div>
      </div>
      <div style={{ width: 1, background: 'rgba(255,255,255,0.1)' }} />
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 6, letterSpacing: '1px' }}>
          MISS DISTANCE
        </div>
        <div style={{ fontSize: 24, fontWeight: 600, color: '#e6edf3', fontFamily: 'monospace' }}>
          {missDistance}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Architecture Overview Component
// ============================================================================

function ArchitectureOverview() {
  const services = [
    { name: 'ingest', port: '8001', desc: 'CubeSat SWIR data acquisition' },
    { name: 'detector', port: '8002', desc: 'YOLOv8 object detection' },
    { name: 'tracker', port: '8003', desc: 'Multi-sensor IOD (planned)' },
    { name: 'propagator', port: '8004', desc: 'Keplerian orbit propagation' },
    { name: 'viz', port: '8005', desc: '3D visualization service' },
    { name: 'ui', port: '8080', desc: 'Web dashboard interface' },
  ];

  return (
    <div style={{ padding: 24 }}>
      <h3 style={{ 
        fontSize: 14, 
        fontWeight: 600, 
        color: '#8b949e', 
        marginBottom: 16,
        letterSpacing: '2px',
      }}>
        MICROSERVICES ARCHITECTURE
      </h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        {services.map((service, i) => (
          <div
            key={service.name}
            style={{
              padding: 16,
              background: 'rgba(0,0,0,0.3)',
              borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.08)',
              transition: 'all 0.3s',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ 
                fontSize: 13, 
                fontWeight: 600, 
                color: '#00d4ff',
                fontFamily: 'monospace',
              }}>
                {service.name}
              </span>
              <span style={{ 
                fontSize: 11, 
                color: '#8b949e',
                fontFamily: 'monospace',
                background: 'rgba(255,255,255,0.05)',
                padding: '2px 8px',
                borderRadius: 4,
              }}>
                :{service.port}
              </span>
            </div>
            <div style={{ fontSize: 11, color: '#8b949e' }}>
              {service.desc}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main Application
// ============================================================================

export default function AveraAtlasDemo() {
  const [activeView, setActiveView] = useState('overview');
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentStage, setCurrentStage] = useState('ingest');
  const [scenarioIndex, setScenarioIndex] = useState(0);

  const stages = ['ingest', 'detect', 'track', 'propagate', 'assess', 'decide'];
  
  // Auto-advance pipeline stages
  useEffect(() => {
    if (!isPlaying || activeView !== 'pipeline') return;
    
    const interval = setInterval(() => {
      setCurrentStage(current => {
        const currentIndex = stages.indexOf(current);
        return stages[(currentIndex + 1) % stages.length];
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [isPlaying, activeView]);

  const scenarios = [
    { riskLevel: 'RED', probability: '2.3e-03', missDistance: '127m' },
    { riskLevel: 'AMBER', probability: '8.1e-05', missDistance: '892m' },
    { riskLevel: 'GREEN', probability: '1.2e-07', missDistance: '4.2km' },
  ];

  // Cycle through scenarios
  useEffect(() => {
    if (!isPlaying) return;
    
    const interval = setInterval(() => {
      setScenarioIndex(i => (i + 1) % scenarios.length);
    }, 5000);

    return () => clearInterval(interval);
  }, [isPlaying]);

  const currentScenario = scenarios[scenarioIndex];

  return (
    <div
      style={{
        minHeight: '100vh',
        background: COLORS.background,
        color: COLORS.text,
        fontFamily: "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      {/* Header */}
      <header
        style={{
          padding: '20px 32px',
          borderBottom: `1px solid ${COLORS.border}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'rgba(0,0,0,0.3)',
        }}
      >
        <div>
          <h1
            style={{
              fontSize: 24,
              fontWeight: 700,
              margin: 0,
              background: 'linear-gradient(135deg, #00d4ff, #7c3aed)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '1px',
            }}
          >
            AVERA-ATLAS
          </h1>
          <p style={{ fontSize: 12, color: '#8b949e', margin: '4px 0 0', letterSpacing: '2px' }}>
            SPACE DEBRIS DETECTION & CONJUNCTION ASSESSMENT
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {['overview', 'orbital', 'tracker', 'pipeline'].map(view => (
            <button
              key={view}
              onClick={() => setActiveView(view)}
              style={{
                padding: '10px 20px',
                background: activeView === view 
                  ? 'linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2))'
                  : 'transparent',
                border: `1px solid ${activeView === view ? '#00d4ff' : COLORS.border}`,
                borderRadius: 8,
                color: activeView === view ? '#00d4ff' : '#8b949e',
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                letterSpacing: '1px',
                textTransform: 'uppercase',
                transition: 'all 0.3s',
              }}
            >
              {view}
            </button>
          ))}
        </div>
      </header>

      {/* Main Content */}
      <main style={{ display: 'flex', height: 'calc(100vh - 81px)' }}>
        {/* Visualization Area */}
        <div style={{ flex: 1, position: 'relative' }}>
          {activeView === 'overview' && (
            <div style={{ padding: 32 }}>
              <div style={{ 
                maxWidth: 900, 
                margin: '0 auto',
              }}>
                <div style={{ marginBottom: 32 }}>
                  <h2 style={{ 
                    fontSize: 32, 
                    fontWeight: 300, 
                    marginBottom: 16,
                    color: '#e6edf3',
                  }}>
                    Analytical Planning System
                  </h2>
                  <p style={{ 
                    fontSize: 16, 
                    color: '#8b949e', 
                    lineHeight: 1.7,
                    maxWidth: 700,
                  }}>
                    AVERA-ATLAS provides real-time space debris tracking and collision risk 
                    assessment through multi-sensor fusion, Keplerian orbital propagation, 
                    and NASA-standard probability calculations.
                  </p>
                </div>
                
                <ArchitectureOverview />
                
                <div style={{ marginTop: 32 }}>
                  <h3 style={{ 
                    fontSize: 14, 
                    fontWeight: 600, 
                    color: '#8b949e', 
                    marginBottom: 16,
                    letterSpacing: '2px',
                  }}>
                    CURRENT ASSESSMENT
                  </h3>
                  <RiskDisplay {...currentScenario} />
                </div>
              </div>
            </div>
          )}
          
          {activeView === 'orbital' && (
            <OrbitalVisualization 
              activeScenario={currentScenario}
              isPlaying={isPlaying}
            />
          )}
          
          {activeView === 'tracker' && (
            <TrackerExplainer isPlaying={isPlaying} />
          )}
          
          {activeView === 'pipeline' && (
            <div style={{ 
              height: '100%', 
              display: 'flex', 
              flexDirection: 'column',
              justifyContent: 'center',
            }}>
              <div style={{ textAlign: 'center', marginBottom: 24 }}>
                <h2 style={{ fontSize: 20, fontWeight: 300, color: '#8b949e', letterSpacing: '3px' }}>
                  APS DATA FLOW
                </h2>
              </div>
              <AnimatedPipeline currentStage={currentStage} />
              <div style={{ 
                display: 'flex', 
                justifyContent: 'center',
                marginTop: 48,
              }}>
                <RiskDisplay {...currentScenario} />
              </div>
            </div>
          )}

          {/* Playback Controls */}
          <div
            style={{
              position: 'absolute',
              bottom: 24,
              left: 24,
              display: 'flex',
              gap: 12,
              alignItems: 'center',
            }}
          >
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              style={{
                width: 48,
                height: 48,
                borderRadius: '50%',
                background: isPlaying 
                  ? 'linear-gradient(135deg, #00d4ff, #7c3aed)'
                  : 'rgba(255,255,255,0.1)',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                transition: 'all 0.3s',
              }}
            >
              {isPlaying ? (
                <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 20, height: 20 }}>
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: 20, height: 20 }}>
                  <polygon points="5,3 19,12 5,21" />
                </svg>
              )}
            </button>
            <span style={{ fontSize: 12, color: '#8b949e' }}>
              {isPlaying ? 'SIMULATION RUNNING' : 'PAUSED'}
            </span>
          </div>
        </div>

        {/* Side Panel */}
        <aside
          style={{
            width: 320,
            borderLeft: `1px solid ${COLORS.border}`,
            padding: 24,
            background: 'rgba(0,0,0,0.2)',
            overflowY: 'auto',
          }}
        >
          <h3 style={{ 
            fontSize: 12, 
            fontWeight: 600, 
            color: '#8b949e', 
            marginBottom: 20,
            letterSpacing: '2px',
          }}>
            SYSTEM STATUS
          </h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[
              { label: 'Ingest Service', status: 'online', color: '#10b981' },
              { label: 'Detector Service', status: 'online', color: '#10b981' },
              { label: 'Tracker Service', status: 'planned', color: '#f59e0b' },
              { label: 'Propagator Service', status: 'online', color: '#10b981' },
              { label: 'Viz Service', status: 'online', color: '#10b981' },
            ].map(service => (
              <div
                key={service.label}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '12px 16px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: 8,
                  border: '1px solid rgba(255,255,255,0.05)',
                }}
              >
                <span style={{ fontSize: 13, color: '#e6edf3' }}>{service.label}</span>
                <span
                  style={{
                    fontSize: 10,
                    color: service.color,
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: service.color,
                      boxShadow: `0 0 8px ${service.color}`,
                    }}
                  />
                  {service.status}
                </span>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 32 }}>
            <h3 style={{ 
              fontSize: 12, 
              fontWeight: 600, 
              color: '#8b949e', 
              marginBottom: 16,
              letterSpacing: '2px',
            }}>
              DEMO OBJECTIVES
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                'Real-time debris detection',
                'Multi-sensor data fusion',
                'Keplerian orbit propagation',
                'Collision probability (Pc)',
                'GO/NO-GO recommendations',
              ].map((objective, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '10px 14px',
                    background: 'rgba(0,212,255,0.05)',
                    borderRadius: 6,
                    border: '1px solid rgba(0,212,255,0.1)',
                  }}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="#00d4ff" strokeWidth="2" style={{ width: 16, height: 16, flexShrink: 0 }}>
                    <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
                    <path d="M22 4L12 14.01l-3-3" />
                  </svg>
                  <span style={{ fontSize: 12, color: '#e6edf3' }}>{objective}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 32 }}>
            <h3 style={{ 
              fontSize: 12, 
              fontWeight: 600, 
              color: '#8b949e', 
              marginBottom: 16,
              letterSpacing: '2px',
            }}>
              ORBITAL VIEW LEGEND
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                { color: '#00d4ff', label: 'ATLAS-1 CubeSat' },
                { color: '#8b5cf6', label: 'ATLAS-2 CubeSat' },
                { color: '#10b981', label: 'ATLAS-3 CubeSat' },
                { color: '#ef4444', label: 'Target Debris' },
                { color: '#6b7280', label: 'Background Debris' },
              ].map(item => (
                <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: 3,
                      background: item.color,
                      boxShadow: `0 0 8px ${item.color}66`,
                    }}
                  />
                  <span style={{ fontSize: 12, color: '#8b949e' }}>{item.label}</span>
                </div>
              ))}
            </div>
            <div style={{ 
              marginTop: 16, 
              padding: 12, 
              background: 'rgba(0,212,255,0.05)', 
              borderRadius: 8,
              border: '1px solid rgba(0,212,255,0.1)',
            }}>
              <div style={{ fontSize: 11, color: '#00d4ff', marginBottom: 6, fontWeight: 600 }}>
                MULTI-SENSOR IOD
              </div>
              <div style={{ fontSize: 11, color: '#8b949e', lineHeight: 1.5 }}>
                Three CubeSats observe the same debris object from different orbital planes, 
                enabling Initial Orbit Determination via Gauss/Herrick-Gibbs methods.
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
