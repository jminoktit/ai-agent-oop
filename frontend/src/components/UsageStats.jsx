import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';

export default function UsageStats() {
  const [stats, setStats] = useState(null);
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (open && !stats) loadStats();
  }, [open]);

  const loadStats = async () => {
    try {
      const data = await api.request('/user-info/');
      setStats({
        username: data.username,
        conversations: 0,
        totalMessages: 0,
        agents: {},
      });
      // Try to get real stats
      try {
        const convData = await api.conversations();
        const convs = convData.conversations || [];
        const agentCounts = {};
        convs.forEach(c => {
          agentCounts[c.agent_name] = (agentCounts[c.agent_name] || 0) + 1;
        });
        setStats(prev => ({
          ...prev,
          conversations: convs.length,
          agents: agentCounts,
        }));
      } catch {}
    } catch {}
  };

  if (!open) {
    return (
      <button className="usage-stats-toggle" onClick={() => setOpen(true)} title="Usage stats">
        📊
      </button>
    );
  }

  const maxAgent = stats?.agents ? Object.entries(stats.agents).sort((a, b) => b[1] - a[1])[0] : null;

  return (
    <div className="modal-overlay" onClick={() => setOpen(false)}>
      <div className="usage-stats-modal" onClick={e => e.stopPropagation()}>
        <div className="usage-stats-header">
          <h3>📊 Usage Statistics</h3>
          <button onClick={() => setOpen(false)}>✕</button>
        </div>

        <div className="usage-stats-grid">
          <div className="usage-stat-card">
            <div className="usage-stat-icon purple">💬</div>
            <div className="usage-stat-info">
              <div className="usage-stat-value">{stats?.conversations || 0}</div>
              <div className="usage-stat-label">Total Conversations</div>
            </div>
          </div>

          <div className="usage-stat-card">
            <div className="usage-stat-icon green">🤖</div>
            <div className="usage-stat-info">
              <div className="usage-stat-value">{Object.keys(stats?.agents || {}).length}</div>
              <div className="usage-stat-label">Agents Used</div>
            </div>
          </div>

          <div className="usage-stat-card">
            <div className="usage-stat-icon yellow">⭐</div>
            <div className="usage-stat-info">
              <div className="usage-stat-value">{maxAgent ? maxAgent[0] : 'N/A'}</div>
              <div className="usage-stat-label">Most Used Agent</div>
            </div>
          </div>

          <div className="usage-stat-card">
            <div className="usage-stat-icon red">🧠</div>
            <div className="usage-stat-info">
              <div className="usage-stat-value">Gemma-2-2B</div>
              <div className="usage-stat-label">Default Model</div>
            </div>
          </div>
        </div>

        {stats?.agents && Object.keys(stats.agents).length > 0 && (
          <div className="usage-agent-breakdown">
            <h4>Agent Usage</h4>
            {Object.entries(stats.agents).sort((a, b) => b[1] - a[1]).map(([agent, count]) => (
              <div key={agent} className="usage-agent-bar">
                <span className="usage-agent-name">{agent}</span>
                <div className="usage-agent-track">
                  <div
                    className="usage-agent-fill"
                    style={{ width: `${(count / stats.conversations) * 100}%` }}
                  />
                </div>
                <span className="usage-agent-count">{count}</span>
              </div>
            ))}
          </div>
        )}

        <div className="usage-stats-actions">
          <button className="modal-btn primary" onClick={() => { setOpen(false); navigate('/training'); }}>
            🧠 Training Dashboard
          </button>
          <button className="modal-btn" onClick={() => setOpen(false)}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
