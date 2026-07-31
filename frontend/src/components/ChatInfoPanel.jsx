import { useState } from 'react';

export default function ChatInfoPanel({ messages, activeAgent, conversationId }) {
  const [open, setOpen] = useState(false);

  const stats = {
    totalMessages: messages.length,
    userMessages: messages.filter(m => m.role === 'user').length,
    assistantMessages: messages.filter(m => m.role === 'assistant').length,
    totalChars: messages.reduce((sum, m) => sum + m.content.length, 0),
    avgMsgLength: messages.length ? Math.round(messages.reduce((sum, m) => sum + m.content.length, 0) / messages.length) : 0,
    codeBlocks: messages.filter(m => m.content.includes('```')).length,
    lastMessage: messages.length ? new Date().toLocaleTimeString() : 'N/A',
  };

  if (!open) {
    return (
      <button className="chat-info-toggle" onClick={() => setOpen(true)} title="Chat info">
        ℹ️
      </button>
    );
  }

  return (
    <div className="chat-info-panel">
      <div className="chat-info-header">
        <h4>Chat Info</h4>
        <button onClick={() => setOpen(false)}>✕</button>
      </div>
      <div className="chat-info-grid">
        <div className="info-item">
          <span className="info-label">Agent</span>
          <span className="info-value">{activeAgent}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Conversation ID</span>
          <span className="info-value">#{conversationId || 'New'}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Total Messages</span>
          <span className="info-value">{stats.totalMessages}</span>
        </div>
        <div className="info-item">
          <span className="info-label">User Messages</span>
          <span className="info-value">{stats.userMessages}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Assistant Messages</span>
          <span className="info-value">{stats.assistantMessages}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Total Characters</span>
          <span className="info-value">{stats.totalChars.toLocaleString()}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Avg Message Length</span>
          <span className="info-value">{stats.avgMsgLength}</span>
        </div>
        <div className="info-item">
          <span className="info-label">Code Blocks</span>
          <span className="info-value">{stats.codeBlocks}</span>
        </div>
      </div>
    </div>
  );
}
