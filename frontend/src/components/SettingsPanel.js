import React, { useState } from 'react';

function SettingsPanel({ settings, updateSettings, onClose }) {
  const [formData, setFormData] = useState({ ...settings });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    updateSettings(formData);
  };

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h3>Settings</h3>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="provider">LLM Provider</label>
          <select 
            id="provider" 
            name="provider" 
            value={formData.provider}
            onChange={handleChange}
          >
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="gemini">Google Gemini</option>
            <option value="custom">Custom API</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="apiKey">API Key</label>
          <input
            type="password"
            id="apiKey"
            name="apiKey"
            value={formData.apiKey}
            onChange={handleChange}
            placeholder="Enter API key (optional)"
          />
          <small>Leave blank to use server's environment variable</small>
        </div>

        <div className="form-group">
          <label htmlFor="modelName">Model Name</label>
          <input
            type="text"
            id="modelName"
            name="modelName"
            value={formData.modelName}
            onChange={handleChange}
            placeholder="Model name"
          />
        </div>

        <div className="form-group checkbox">
          <input
            type="checkbox"
            id="showSourcesInResponse"
            name="showSourcesInResponse"
            checked={formData.showSourcesInResponse}
            onChange={handleChange}
          />
          <label htmlFor="showSourcesInResponse">Include sources in response</label>
        </div>

        <div className="form-actions">
          <button type="submit">Save Settings</button>
          <button type="button" onClick={onClose}>Cancel</button>
        </div>
      </form>
    </div>
  );
}

export default SettingsPanel;