// Compare.jsx — All-model comparison table + Recharts grouped bar chart
import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { getMetrics } from '../api';
import Loading from '../components/Loading';
import ErrorMessage from '../components/ErrorMessage';

const METRICS_KEYS = ['accuracy', 'precision', 'recall', 'f1'];
const COLORS = ['#14f0c8', '#7c6fff', '#fbbf24', '#ff6b8a'];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(17,24,39,0.95)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '10px',
      padding: '0.75rem 1rem',
      fontSize: '0.82rem',
    }}>
      <p style={{ fontWeight: 700, marginBottom: '0.4rem', color: 'var(--text-primary)' }}>{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.fill, margin: '0.15rem 0' }}>
          {p.name}: {(p.value * 100).toFixed(1)}%
        </p>
      ))}
    </div>
  );
};

export default function Compare() {
  const [raw, setRaw] = useState(null);
  const [error, setError] = useState(null);

  const load = () => {
    setError(null);
    getMetrics().then(setRaw).catch(e => setError(e.message));
  };

  useEffect(() => { load(); }, []);

  // Shape data for Recharts: [{name: "LogReg", accuracy: 0.80, ...}, ...]
  const chartData = raw
    ? Object.entries(raw).map(([key, m]) => ({
        name: m.display_name ?? key,
        accuracy: m.accuracy,
        precision: m.precision,
        recall: m.recall,
        f1: m.f1,
        is_active: m.is_active,
      }))
    : [];

  return (
    <main className="page">
      <div className="section-header" style={{ marginBottom: '0.5rem' }}>
        <h1>Algorithm <span className="gradient-text">Comparison</span></h1>
        <span className="section-badge badge-amber">4 Models</span>
      </div>
      <p className="subtitle" style={{ marginBottom: '2rem' }}>
        Side-by-side comparison of all trained models across four key classification metrics.
        The active production model is highlighted.
      </p>

      {error && <ErrorMessage message={error} onRetry={load} />}
      {!raw && !error && <Loading text="Loading all model metrics..." />}

      {raw && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

          {/* Comparison table */}
          <div className="data-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(raw).map(([key, m]) => (
                  <tr key={key} className={m.is_active ? 'active-row' : ''}>
                    <td style={{ fontWeight: m.is_active ? 700 : 400, color: m.is_active ? 'var(--accent-teal)' : 'var(--text-secondary)' }}>
                      {m.display_name ?? key}
                    </td>
                    <td>{(m.accuracy * 100).toFixed(1)}%</td>
                    <td>{(m.precision * 100).toFixed(1)}%</td>
                    <td>{(m.recall * 100).toFixed(1)}%</td>
                    <td>{(m.f1 * 100).toFixed(1)}%</td>
                    <td>
                      {m.is_active
                        ? <span className="section-badge badge-teal">✓ Deployed</span>
                        : <span className="section-badge" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-muted)' }}>Comparison</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Recharts grouped bar chart */}
          <div className="card">
            <h2 style={{ marginBottom: '1.5rem' }}>📊 Metric Comparison Chart</h2>
            <ResponsiveContainer width="100%" height={360}>
              <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }} barGap={4} barCategoryGap="28%">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#8b9dc3', fontSize: 12, fontFamily: 'Inter' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                  tick={{ fill: '#8b9dc3', fontSize: 11, fontFamily: 'Inter' }}
                  axisLine={false}
                  tickLine={false}
                  width={44}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Legend
                  wrapperStyle={{ fontSize: '0.82rem', color: '#8b9dc3', fontFamily: 'Inter', paddingTop: '1rem' }}
                />
                {METRICS_KEYS.map((key, i) => (
                  <Bar
                    key={key}
                    dataKey={key}
                    name={key.charAt(0).toUpperCase() + key.slice(1)}
                    fill={COLORS[i]}
                    radius={[4, 4, 0, 0]}
                    maxBarSize={40}
                    fillOpacity={0.85}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>

        </div>
      )}
    </main>
  );
}
