export default function TypingIndicator({ agentName = 'assistant', agentIcon = '🤖' }) {
  return (
    <div className="message assistant">
      <div className="message-avatar">{agentIcon}</div>
      <div className="message-body">
        <div className="typing-bubble">
          <div className="typing-text">{agentName} is thinking</div>
          <div className="typing-dots">
            <span className="typing-dot"></span>
            <span className="typing-dot"></span>
            <span className="typing-dot"></span>
          </div>
        </div>
      </div>
    </div>
  );
}
