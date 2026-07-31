import { useState, useEffect } from 'react';

export default function ConversationTimer({ startedAt }) {
  const [elapsed, setElapsed] = useState('00:00');

  useEffect(() => {
    if (!startedAt) return;
    const start = new Date(startedAt).getTime();

    const update = () => {
      const diff = Math.floor((Date.now() - start) / 1000);
      const mins = Math.floor(diff / 60).toString().padStart(2, '0');
      const secs = (diff % 60).toString().padStart(2, '0');
      setElapsed(`${mins}:${secs}`);
    };

    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, [startedAt]);

  return (
    <span className="conversation-timer">
      <span className="timer-dot"></span>
      {elapsed}
    </span>
  );
}
