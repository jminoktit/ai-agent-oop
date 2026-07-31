import { useState, useMemo } from 'react';

export default function MessageSearch({ messages, onJumpTo }) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);

  const results = useMemo(() => {
    if (!query.trim()) return [];
    const q = query.toLowerCase();
    return messages
      .map((m, i) => ({ ...m, index: i }))
      .filter(m => m.content.toLowerCase().includes(q));
  }, [query, messages]);

  const highlight = (text, q) => {
    if (!q.trim()) return text;
    const regex = new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);
    return parts.map((part, i) =>
      regex.test(part) ? <mark key={i} className="search-highlight">{part}</mark> : part
    );
  };

  if (!open) {
    return (
      <button className="search-toggle-btn" onClick={() => setOpen(true)} title="Search messages">
        🔍
      </button>
    );
  }

  return (
    <div className="message-search-panel">
      <div className="message-search-bar">
        <input
          type="text"
          placeholder="Search in conversation..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          autoFocus
          className="message-search-input"
        />
        <span className="search-count">{results.length} found</span>
        <button className="search-close-btn" onClick={() => { setOpen(false); setQuery(''); }}>✕</button>
      </div>
      {query && (
        <div className="message-search-results">
          {results.length === 0 ? (
            <div className="search-no-results">No results found</div>
          ) : (
            results.map(r => (
              <div
                key={r.index}
                className="search-result-item"
                onClick={() => { onJumpTo(r.index); setOpen(false); setQuery(''); }}
              >
                <span className={`search-result-role ${r.role}`}>{r.role === 'user' ? '👤' : '🤖'}</span>
                <span className="search-result-text">
                  {highlight(r.content.slice(0, 120), query)}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
