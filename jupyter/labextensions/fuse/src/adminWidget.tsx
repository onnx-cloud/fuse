import React, { useEffect, useState } from 'react';

export function AdminWidget(){
  const [engines, setEngines] = useState<any>({});
  const [status, setStatus] = useState('');
  const [statusType, setStatusType] = useState<'success'|'error'|'info'>('info');
  const [editing, setEditing] = useState<any>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [testing, setTesting] = useState(false);

  useEffect(()=>{ (async ()=>{ try{ const r = await fetch('/fuse/api/llm/admin'); if(!r.ok){ showStatus('admin disabled or no permission', 'error'); return;} const j = await r.json(); setEngines(j); }catch(e){ showStatus('failed to load '+String(e), 'error'); } })(); }, []);

  const showStatus = (msg: string, type: 'success'|'error'|'info' = 'info') => {
    setStatus(msg);
    setStatusType(type);
    setTimeout(() => setStatus(''), 5000);
  };

  const validateEngine = (eng: any): string[] => {
    const errors: string[] = [];
    if (!eng.name || eng.name.trim() === '') errors.push('Name is required');
    if (eng.name && !/^[a-zA-Z0-9_-]+$/.test(eng.name)) errors.push('Name must be alphanumeric with - or _');
    if (!eng.url || eng.url.trim() === '') errors.push('URL is required');
    if (eng.url && !eng.url.startsWith('http://') && !eng.url.startsWith('https://')) errors.push('URL must start with http:// or https://');
    if (!eng.model || eng.model.trim() === '') errors.push('Model is required');
    return errors;
  };

  const startCreate = ()=>{
    setEditing({ name: '', url:'', secretEnv:'', label:'', model:'', prompt:'You are a helpful assistant.' });
    setValidationErrors([]);
    setStatus('');
  };

  const startEdit = (name:string)=>{
    const cfg = engines[name] || {};
    setEditing({ name, url: cfg.url || '', secretEnv: cfg.secretEnv || '', label: cfg.label || name, model: cfg.model || '', prompt: cfg.prompt || '' });
    setValidationErrors([]);
    setStatus('');
  };

  const saveEngine = async ()=>{
    if(!editing) return;
    const errors = validateEngine(editing);
    if (errors.length > 0) {
      setValidationErrors(errors);
      showStatus('Please fix validation errors', 'error');
      return;
    }
    const name = editing.name;
    const body = { url: editing.url, secretEnv: editing.secretEnv, label: editing.label, model: editing.model, prompt: editing.prompt };
    try {
      const r = await fetch('/fuse/api/llm/admin/'+encodeURIComponent(name), {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
      if(r.ok){ 
        showStatus('✓ Saved successfully', 'success');
        setEngines({...engines, [name]: body}); 
        setEditing(null);
        setValidationErrors([]);
      } else { 
        showStatus('Failed: '+String(await r.text()), 'error');
      }
    } catch (e) {
      showStatus('Network error: '+String(e), 'error');
    }
  };

  const deleteEngine = async (name:string)=>{
    if(!confirm('Delete engine '+name+'?')) return;
    const r = await fetch('/fuse/api/llm/admin/'+encodeURIComponent(name), {method: 'DELETE'});
    if(r.ok){ const copy = {...engines}; delete copy[name]; setEngines(copy); showStatus('✓ Deleted ' + name, 'success');} else { showStatus('Delete failed', 'error'); }
  };

  const testConnection = async ()=>{
    if(!editing) return;
    const errors = validateEngine(editing);
    if (errors.length > 0) {
      setValidationErrors(errors);
      showStatus('Fix validation errors before testing', 'error');
      return;
    }
    setTesting(true);
    showStatus('Testing connection...', 'info');
    try{
      const r = await fetch('/fuse/api/llm', {
        method: 'POST', 
        headers: {'Content-Type':'application/json'}, 
        body: JSON.stringify({
          engine: editing.name || '__test__', 
          messages:[{role:'user', content:'Hello'}], 
          stream:false
        })
      });
      if (r.ok) {
        const j = await r.json();
        const content = j?.choices?.[0]?.message?.content;
        if (content) {
          showStatus('✓ Connection successful! Response: ' + content.substring(0, 50), 'success');
        } else {
          showStatus('Connection OK but unexpected response format', 'info');
        }
      } else {
        showStatus('Connection failed: HTTP ' + r.status, 'error');
      }
    }catch(e){ 
      showStatus('Connection failed: '+String(e), 'error');
    } finally {
      setTesting(false);
    }
  };

  const exportConfig = () => {
    const json = JSON.stringify(engines, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'fuse-llm-config.json';
    a.click();
    URL.revokeObjectURL(url);
    showStatus('✓ Config exported', 'success');
  };

  const importConfig = async (event: any) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const imported = JSON.parse(text);
      if (typeof imported !== 'object' || Array.isArray(imported)) {
        showStatus('Invalid config format', 'error');
        return;
      }
      setEngines(imported);
      showStatus('✓ Config imported (' + Object.keys(imported).length + ' engines)', 'success');
    } catch (e) {
      showStatus('Import failed: ' + String(e), 'error');
    }
  };

  const statusStyle = {
    padding: '12px',
    borderRadius: '4px',
    marginBottom: '12px',
    backgroundColor: statusType === 'success' ? '#d4edda' : statusType === 'error' ? '#f8d7da' : '#d1ecf1',
    color: statusType === 'success' ? '#155724' : statusType === 'error' ? '#721c24' : '#0c5460',
    border: `1px solid ${statusType === 'success' ? '#c3e6cb' : statusType === 'error' ? '#f5c6cb' : '#bee5eb'}`
  };

  const cardStyle = {
    border: '1px solid #ddd',
    borderRadius: '8px',
    padding: '16px',
    marginBottom: '12px',
    backgroundColor: '#f9f9f9'
  };

  const inputStyle = {
    width: '100%',
    padding: '8px',
    marginTop: '4px',
    marginBottom: '12px',
    border: '1px solid #ccc',
    borderRadius: '4px'
  };

  const buttonStyle = {
    padding: '8px 16px',
    marginRight: '8px',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    backgroundColor: '#007bff',
    color: 'white'
  };

  const secondaryButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#6c757d'
  };

  const dangerButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#dc3545'
  };

  return React.createElement('div', { style: { maxWidth: '800px', margin: '0 auto' } },
    React.createElement('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' } },
      React.createElement('h2', { style: { margin: 0 } }, 'LLM Engine Management'),
      React.createElement('div', {},
        React.createElement('button', { onClick: exportConfig, style: secondaryButtonStyle }, '📥 Export'),
        React.createElement('label', { style: { ...secondaryButtonStyle, display: 'inline-block' } },
          '📤 Import',
          React.createElement('input', { type: 'file', accept: '.json', onChange: importConfig, style: { display: 'none' } })
        )
      )
    ),
    
    status ? React.createElement('div', { style: statusStyle }, status) : null,
    
    editing ? React.createElement('div', { style: cardStyle },
      React.createElement('h3', { style: { marginTop: 0 } }, editing.name ? 'Edit Engine: ' + editing.name : 'New Engine'),
      
      validationErrors.length > 0 ? React.createElement('div', { style: { ...statusStyle, backgroundColor: '#f8d7da', color: '#721c24', border: '1px solid #f5c6cb' } },
        React.createElement('strong', {}, '⚠️ Validation Errors:'),
        React.createElement('ul', { style: { margin: '8px 0 0 0' } }, validationErrors.map((err, i) => React.createElement('li', { key: i }, err)))
      ) : null,
      
      React.createElement('div', {},
        React.createElement('label', { style: { fontWeight: 'bold' } }, 'Name *'),
        React.createElement('input', { 
          style: inputStyle,
          placeholder: 'e.g., openai, anthropic',
          value: editing.name, 
          onChange: (ev:any)=> { setEditing({...editing, name: ev.target.value}); setValidationErrors([]); }
        })
      ),
      
      React.createElement('div', {},
        React.createElement('label', { style: { fontWeight: 'bold' } }, 'URL *'),
        React.createElement('input', { 
          style: inputStyle,
          placeholder: 'https://api.openai.com/v1/chat/completions',
          value: editing.url, 
          onChange: (ev:any)=> { setEditing({...editing, url: ev.target.value}); setValidationErrors([]); }
        })
      ),
      
      React.createElement('div', {},
        React.createElement('label', { style: { fontWeight: 'bold' } }, 'Model *'),
        React.createElement('input', { 
          style: inputStyle,
          placeholder: 'gpt-4, claude-3-opus',
          value: editing.model, 
          onChange: (ev:any)=> { setEditing({...editing, model: ev.target.value}); setValidationErrors([]); }
        })
      ),
      
      React.createElement('div', {},
        React.createElement('label', { style: { fontWeight: 'bold' } }, 'Secret Environment Variable'),
        React.createElement('input', { 
          style: inputStyle,
          placeholder: 'OPENAI_API_KEY',
          value: editing.secretEnv, 
          onChange: (ev:any)=> setEditing({...editing, secretEnv: ev.target.value})
        })
      ),
      
      React.createElement('div', {},
        React.createElement('label', { style: { fontWeight: 'bold' } }, 'Display Label'),
        React.createElement('input', { 
          style: inputStyle,
          placeholder: 'OpenAI GPT-4',
          value: editing.label, 
          onChange: (ev:any)=> setEditing({...editing, label: ev.target.value})
        })
      ),
      
      React.createElement('div', {},
        React.createElement('label', { style: { fontWeight: 'bold' } }, 'System Prompt'),
        React.createElement('textarea', { 
          style: { ...inputStyle, minHeight: '80px', fontFamily: 'monospace' },
          placeholder: 'You are a helpful assistant specialized in ONNX and Fuse.',
          value: editing.prompt, 
          onChange: (ev:any)=> setEditing({...editing, prompt: ev.target.value})
        })
      ),
      
      React.createElement('div', { style: { marginTop: '16px' } },
        React.createElement('button', { onClick: saveEngine, style: buttonStyle }, '💾 Save'),
        React.createElement('button', { onClick: ()=> { setEditing(null); setValidationErrors([]); }, style: secondaryButtonStyle }, 'Cancel'),
        React.createElement('button', { 
          onClick: testConnection, 
          style: { ...buttonStyle, backgroundColor: testing ? '#6c757d' : '#28a745' },
          disabled: testing
        }, testing ? '⏳ Testing...' : '🔌 Test Connection')
      )
    ) : React.createElement('button', { onClick: startCreate, style: buttonStyle }, '➕ Add Engine'),
    
    React.createElement('div', { style: { marginTop: '24px' } },
      React.createElement('h3', {}, 'Configured Engines (' + Object.keys(engines).length + ')'),
      Object.keys(engines).length === 0 ? React.createElement('p', { style: { color: '#666', fontStyle: 'italic' } }, 'No engines configured. Click "Add Engine" to get started.') : null,
      React.createElement('div', {}, Object.entries(engines).map(([k,v]:any)=>React.createElement('div', { 
        key: k,
        style: {
          border: '1px solid #e0e0e0',
          borderRadius: '6px',
          padding: '12px',
          marginBottom: '8px',
          backgroundColor: 'white',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }
      },
        React.createElement('div', {},
          React.createElement('strong', { style: { fontSize: '16px' } }, k),
          React.createElement('div', { style: { color: '#666', fontSize: '14px', marginTop: '4px' } }, 
            v.label || k,
            ' • ',
            v.model || 'no model'
          ),
          React.createElement('div', { style: { color: '#999', fontSize: '12px', marginTop: '2px' } }, v.url || '')
        ),
        React.createElement('div', {},
          React.createElement('button', { onClick: ()=> startEdit(k), style: buttonStyle }, '✏️ Edit'),
          React.createElement('button', { onClick: ()=> deleteEngine(k), style: dangerButtonStyle }, '🗑️ Delete')
        )
      )))
    )
  );
}

export function AdminWidgetComponent(){
  return React.createElement('div', {}, React.createElement(AdminWidget));
}
