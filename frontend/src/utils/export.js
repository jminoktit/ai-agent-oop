export function exportAsMarkdown(messages, title = 'Chat Export') {
  let md = `# ${title}\n\n`;
  md += `*Exported on ${new Date().toLocaleString()}*\n\n---\n\n`;

  messages.forEach((msg) => {
    const role = msg.role === 'user' ? '**You**' : '**Assistant**';
    md += `### ${role}\n\n${msg.content}\n\n---\n\n`;
  });

  const blob = new Blob([md], { type: 'text/markdown' });
  downloadBlob(blob, `${sanitizeFilename(title)}.md`);
}

export function exportAsJson(messages, title = 'Chat Export') {
  const data = {
    title,
    exportedAt: new Date().toISOString(),
    messages: messages.map((m) => ({
      role: m.role,
      content: m.content,
    })),
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  downloadBlob(blob, `${sanitizeFilename(title)}.json`);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function sanitizeFilename(name) {
  return name.replace(/[^a-z0-9]/gi, '_').toLowerCase().slice(0, 50);
}
