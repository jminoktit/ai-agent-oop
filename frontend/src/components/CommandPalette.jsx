import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

const COMMANDS = [
  { id: 'new-chat', icon: '💬', label: 'New Chat', action: 'newChat', category: 'Chat' },
  { id: 'clear-chat', icon: '🗑️', label: 'Clear Chat', action: 'clearChat', category: 'Chat' },
  { id: 'switch-chat', icon: '💬', label: 'Switch to ChatBot', action: 'switchChat', category: 'Agents', agent: 'chat' },
  { id: 'switch-code', icon: '💻', label: 'Switch to CodeBot', action: 'switchAgent', category: 'Agents', agent: 'code' },
  { id: 'switch-data', icon: '📊', label: 'Switch to DataBot', action: 'switchAgent', category: 'Agents', agent: 'data' },
  { id: 'switch-research', icon: '🔍', label: 'Switch to ResearchBot', action: 'switchAgent', category: 'Agents', agent: 'research' },
  { id: 'switch-planner', icon: '📋', label: 'Switch to PlannerBot', action: 'switchAgent', category: 'Agents', agent: 'planner' },
  { id: 'switch-media', icon: '🎨', label: 'Switch to MediaBot', action: 'switchAgent', category: 'Agents', agent: 'media' },
  { id: 'training', icon: '🧠', label: 'Go to Training', action: 'gotoTraining', category: 'Navigation' },
  { id: 'settings', icon: '⚙️', label: 'Go to Settings', action: 'gotoSettings', category: 'Navigation' },
  { id: 'export-md', icon: '📄', label: 'Export as Markdown', action: 'exportMd', category: 'Export' },
  { id: 'export-json', icon: '📦', label: 'Export as JSON', action: 'exportJson', category: 'Export' },
];

export default function CommandPalette({ open, onClose, onCommand }) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef(null);

  useEffect(() => {
    if (open) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  const filtered = COMMANDS.filter((cmd) =>
    cmd.label.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filtered[selectedIndex]) {
        onCommand(filtered[selectedIndex]);
        onClose();
      }
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!open) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="command-palette" onClick={(e) => e.stopPropagation()}>
        <div className="command-search">
          <span className="command-search-icon">🔍</span>
          <input
            ref={inputRef}
            className="command-input"
            placeholder="Type a command..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <span className="command-shortcut">ESC</span>
        </div>
        <div className="command-list">
          {filtered.length === 0 ? (
            <div className="command-empty">No commands found</div>
          ) : (
            filtered.map((cmd, i) => (
              <div
                key={cmd.id}
                className={`command-item ${i === selectedIndex ? 'selected' : ''}`}
                onClick={() => { onCommand(cmd); onClose(); }}
                onMouseEnter={() => setSelectedIndex(i)}
              >
                <span className="command-icon">{cmd.icon}</span>
                <span className="command-label">{cmd.label}</span>
                <span className="command-category">{cmd.category}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
