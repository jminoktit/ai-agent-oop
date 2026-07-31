const TIPS = [
  { icon: '💡', title: 'Be specific', desc: 'The more details you provide, the better the response.' },
  { icon: '📎', title: 'Drag & drop files', desc: 'Upload documents, images, or code files instantly.' },
  { icon: '⌨️', title: 'Use shortcuts', desc: 'Press Ctrl+K for commands, Ctrl+P for templates.' },
  { icon: '🎤', title: 'Voice input', desc: 'Click the mic icon to speak instead of type.' },
];

const SUGGESTIONS = [
  { icon: '💻', text: 'Write a Python web scraper for news articles', agent: 'code' },
  { icon: '📊', text: 'Analyze sales data and create a summary report', agent: 'data' },
  { icon: '🔍', text: 'Research the latest developments in quantum computing', agent: 'research' },
  { icon: '📋', text: 'Create a 3-month project plan for a mobile app', agent: 'planner' },
];

export default function WelcomeScreen({ onSuggestion, onTemplates }) {
  return (
    <div className="welcome-screen">
      <div className="welcome-hero">
        <div className="welcome-logo-anim">
          <div className="welcome-logo-ring"></div>
          <div className="welcome-logo-ring ring-2"></div>
          <div className="welcome-logo-core">AB</div>
        </div>
        <h2 className="welcome-title">Welcome to Aura Book</h2>
        <p className="welcome-subtitle">Your intelligent AI agent platform. Choose an agent or start typing.</p>
      </div>

      <div className="welcome-suggestions">
        <h3>Try asking:</h3>
        <div className="suggestion-grid">
          {SUGGESTIONS.map(s => (
            <button key={s.text} className="suggestion-card" onClick={() => onSuggestion?.(s.text, s.agent)}>
              <span className="suggestion-icon">{s.icon}</span>
              <span className="suggestion-text">{s.text}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="welcome-tips">
        {TIPS.map(tip => (
          <div key={tip.title} className="welcome-tip">
            <span className="tip-icon">{tip.icon}</span>
            <div>
              <div className="tip-title">{tip.title}</div>
              <div className="tip-desc">{tip.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <button className="prompt-templates-cta" onClick={onTemplates}>
        📝 Browse 16 Prompt Templates
      </button>
    </div>
  );
}
