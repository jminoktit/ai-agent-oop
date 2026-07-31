import { useState, useEffect } from 'react';

export default function JumpToBottom({ containerRef, targetRef }) {
  const [show, setShow] = useState(false);

  useEffect(() => {
    const container = containerRef?.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const distFromBottom = scrollHeight - scrollTop - clientHeight;
      setShow(distFromBottom > 200);
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [containerRef]);

  const scrollToBottom = () => {
    targetRef?.current?.scrollIntoView({ behavior: 'smooth' });
  };

  if (!show) return null;

  return (
    <button className="jump-to-bottom" onClick={scrollToBottom} title="Jump to latest message">
      <span className="jump-to-bottom-icon">↓</span>
      <span className="jump-to-bottom-text">Latest</span>
    </button>
  );
}
