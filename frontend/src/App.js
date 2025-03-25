import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';
import SourcePanel from './components/SourcePanel';
import SettingsPanel from './components/SettingsPanel';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const [showSources, setShowSources] = useState(false);
  const [settings, setSettings] = useState({
    provider: 'openai',
    apiKey: '',
    modelName: 'gpt-4-turbo',
    showSourcesInResponse: true
  });
  const [showSettings, setShowSettings] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const messagesEndRef = useRef(null);

  // Fetch system status on load
  useEffect(() => {
    fetchSystemStatus();
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Error fetching system status:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = { text: input, sender: 'user', timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setSources([]);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
          provider: settings.provider,
          api_key: settings.apiKey || undefined,
          model_name: settings.modelName,
          show_sources: settings.showSourcesInResponse
        }),
      });

      const data = await response.json();
      
      if (data.error) {
        setMessages(prev => [...prev, { 
          text: `Error: ${data.error}`, 
          sender: 'system', 
          timestamp: new Date() 
        }]);
      } else {
        // Add bot message
        setMessages(prev => [...prev, { 
          text: data.response, 
          sender: 'bot', 
          timestamp: new Date(),
          processingTime: data.processing_time
        }]);

        // Set sources if available
        if (data.sources && data.sources.length > 0) {
          setSources(data.sources);
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        text: `Error: ${error.message}`, 
        sender: 'system', 
        timestamp: new Date() 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleIngest = async (path) => {
    try {
      setIsLoading(true);
      setMessages(prev => [...prev, { 
        text: `Ingesting documents from: ${path}`, 
        sender: 'system', 
        timestamp: new Date() 
      }]);

      const response = await fetch('/api/ingest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ path }),
      });

      const data = await response.json();
      
      if (data.error) {
        setMessages(prev => [...prev, { 
          text: `Error: ${data.error}`, 
          sender: 'system', 
          timestamp: new Date() 
        }]);
      } else {
        setMessages(prev => [...prev, { 
          text: data.message, 
          sender: 'system', 
          timestamp: new Date(),
          processingTime: data.processing_time
        }]);
        
        // Refresh system status
        fetchSystemStatus();
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        text: `Error: ${error.message}`, 
        sender: 'system', 
        timestamp: new Date() 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleSources = () => {
    setShowSources(!showSources);
  };

  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };

  const updateSettings = (newSettings) => {
    setSettings(newSettings);
    setShowSettings(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG Chatbot</h1>
        <div className="header-buttons">
          <button onClick={toggleSettings}>Settings</button>
          {sources.length > 0 && (
            <button onClick={toggleSources}>
              {showSources ? 'Hide Sources' : 'Show Sources'}
            </button>
          )}
        </div>
      </header>

      <div className="main-container">
        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h2>Welcome to RAG Chatbot</h2>
                <p>This system uses Retrieval-Augmented Generation to answer your questions based on ingested documents.</p>
                {systemStatus && (
                  <div className="system-status">
                    <h3>System Status</h3>
                    <p>Documents in index: {systemStatus.index_size}</p>
                    <p>Embedding dimension: {systemStatus.embedding_dimension}</p>
                    <p>Chunk size: {systemStatus.chunk_size}</p>
                    <p>Retrieval count: {systemStatus.top_k}</p>
                  </div>
                )}
                <div className="ingest-panel">
                  <h3>Ingest Documents</h3>
                  <p>Enter a file or directory path to ingest:</p>
                  <div className="ingest-form">
                    <input
                      type="text"
                      id="ingestPath"
                      placeholder="/path/to/documents"
                    />
                    <button onClick={() => handleIngest(document.getElementById('ingestPath').value)}>
                      Ingest
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))}
            
            {isLoading && (
              <div className="message system">
                <div className="loading-indicator">
                  <div className="loading-spinner"></div>
                  <span>Processing...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !input.trim()}>
              Send
            </button>
          </form>
        </div>

        {showSources && sources.length > 0 && (
          <SourcePanel sources={sources} />
        )}

        {showSettings && (
          <SettingsPanel 
            settings={settings} 
            updateSettings={updateSettings} 
            onClose={() => setShowSettings(false)} 
          />
        )}
      </div>
    </div>
  );
}

export default App;