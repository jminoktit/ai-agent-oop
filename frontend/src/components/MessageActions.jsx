import { useState } from 'react';

export default function MessageActions({ content, isUser, onRegenerate }) {
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState(null); // null, 'up', 'down'

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = content;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return (
      <div className="message-actions">
        <button className="msg-action-btn" onClick={handleCopy} title="Copy">
          {copied ? '✓' : '📋'}
        </button>
      </div>
    );
  }

  return (
    <div className="message-actions">
      <button className="msg-action-btn" onClick={handleCopy} title="Copy">
        {copied ? '✓' : '📋'}
      </button>
      {onRegenerate && (
        <button className="msg-action-btn" onClick={onRegenerate} title="Regenerate">
          🔄
        </button>
      )}
      <button
        className={`msg-action-btn ${liked === 'up' ? 'active' : ''}`}
        onClick={() => setLiked(liked === 'up' ? null : 'up')}
        title="Good response"
      >
        👍
      </button>
      <button
        className={`msg-action-btn ${liked === 'down' ? 'active' : ''}`}
        onClick={() => setLiked(liked === 'down' ? null : 'down')}
        title="Bad response"
      >
        👎
      </button>
    </div>
  );
}
