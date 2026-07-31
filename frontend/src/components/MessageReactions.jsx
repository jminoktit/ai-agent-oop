import { useState } from 'react';

const REACTIONS = ['👍', '❤️', '😄', '🤔', '👎', '🔥', '💡', '✅'];

export default function MessageReactions({ reactions = {}, onReact }) {
  const [showPicker, setShowPicker] = useState(false);
  const activeReactions = Object.entries(reactions).filter(([, count]) => count > 0);

  return (
    <div className="message-reactions">
      {activeReactions.map(([emoji, count]) => (
        <button
          key={emoji}
          className={`reaction-chip ${reactions[`_user_${emoji}`] ? 'active' : ''}`}
          onClick={() => onReact?.(emoji)}
        >
          {emoji} {count}
        </button>
      ))}
      <div className="reaction-add-wrapper">
        <button className="reaction-add-btn" onClick={() => setShowPicker(!showPicker)}>+</button>
        {showPicker && (
          <div className="reaction-picker">
            {REACTIONS.map(emoji => (
              <button
                key={emoji}
                className="reaction-picker-btn"
                onClick={() => { onReact?.(emoji); setShowPicker(false); }}
              >
                {emoji}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
