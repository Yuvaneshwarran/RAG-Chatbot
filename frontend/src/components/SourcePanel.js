import React from 'react';

function SourcePanel({ sources }) {
  return (
    <div className="source-panel">
      <h3>Sources</h3>
      <div className="sources-list">
        {sources.map((source, index) => (
          <div key={index} className="source-item">
            <div className="source-header">
              <span className="source-number">Source {index + 1}</span>
              <span className="source-score">Score: {source.score.toFixed(4)}</span>
            </div>
            {source.metadata && source.metadata.source && (
              <div className="source-file">
                File: {source.metadata.source}
              </div>
            )}
            <div className="source-text">
              {source.text}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SourcePanel;