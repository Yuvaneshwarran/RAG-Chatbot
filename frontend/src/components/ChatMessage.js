import React from 'react';
import ReactMarkdown from 'react-markdown';
import { formatDistanceToNow } from 'date-fns';

function ChatMessage({ message }) {
  const { text, sender, timestamp, processingTime } = message;
  
  return (
    <div className={`message ${sender}`}>
      <div className="message-header">
        <span className="sender">{sender === 'user' ? 'You' : sender === 'bot' ? 'AI' : 'System'}</span>
        <span className="timestamp">{formatDistanceToNow(new Date(timestamp), { addSuffix: true })}</span>
      </div>
      <div className="message-content">
        <ReactMarkdown>{text}</ReactMarkdown>
      </div>
      {processingTime && (
        <div className="message-footer">
          <span className="processing-time">Processed in {processingTime.toFixed(2)}s</span>
        </div>
      )}
    </div>
  );
}

export default ChatMessage;