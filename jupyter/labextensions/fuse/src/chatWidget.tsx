import React from 'react';

// Imperative builder: returns a DOM element with full chat UI
export function createChatWidgetElement() {
  const wrapper = document.createElement('div');
  wrapper.style.display = 'flex';
  wrapper.style.flexDirection = 'column';
  wrapper.style.height = '100%';

  const log = document.createElement('div');
  log.id = 'fuse-chat-log';
  log.style.flex = '1';
  log.style.overflowY = 'auto';
  log.style.padding = '8px';
  log.appendChild(document.createTextNode('Fuse Copilot — ready'));

  const controls = document.createElement('div');
  controls.style.padding = '8px';
  controls.style.borderTop = '1px solid #eee';

  const select = document.createElement('select');
  select.id = 'fuse-chat-engine';
  select.style.marginRight = '8px';

  const input = document.createElement('textarea');
  input.id = 'fuse-chat-input';
  input.placeholder = 'Say something... Shift+Enter for newline';
  input.style.width = '60%';
  input.style.resize = 'vertical';

  const send = document.createElement('button');
  send.id = 'fuse-chat-send';
  send.textContent = 'Send';

  const streamToggle = document.createElement('input');
  streamToggle.type = 'checkbox';
  streamToggle.id = 'fuse-chat-stream';
  const streamLabel = document.createElement('label');
  streamLabel.htmlFor = 'fuse-chat-stream';
  streamLabel.textContent = 'Stream';
  streamLabel.style.marginLeft = '8px';

  // additional controls: clear, export
  const clearBtn = document.createElement('button'); clearBtn.textContent = 'Clear'; clearBtn.className = 'fuse-chat-small';
  const exportBtn = document.createElement('button'); exportBtn.textContent = 'Export'; exportBtn.className = 'fuse-chat-small';

  controls.appendChild(select);
  controls.appendChild(input);
  controls.appendChild(send);
  controls.appendChild(streamToggle);
  controls.appendChild(streamLabel);
  controls.appendChild(clearBtn);
  controls.appendChild(exportBtn);

  async function renderMessage(content: string, isUser: boolean, insertCallback?: (text: string) => void) {
    let html = '';
    try {
      const md = (await import('marked')) as any;
      const dm = (await import('dompurify')) as any;
      
      // Configure marked with syntax highlighting
      if (md?.marked?.setOptions) {
        md.marked.setOptions({
          highlight: (code: string, lang: string) => {
            // Simple syntax highlighting via CSS classes
            return `<code class="language-${lang || 'plaintext'}">${code}</code>`;
          }
        });
      }
      
      const parsed = md?.marked ? md.marked.parse(content) : (md?.parse ? md.parse(content) : String(content));
      const sanitized = dm?.default ? dm.default.sanitize(parsed) : (dm?.sanitize ? dm.sanitize(parsed) : parsed);
      html = sanitized;
    } catch (e) {
      html = String(content).replace(/\n/g, '<br>');
    }
    
    const line = document.createElement('div'); line.className = 'fuse-chat-line';
    const bubble = document.createElement('div');
    bubble.className = isUser ? 'fuse-chat-bubble-user' : 'fuse-chat-bubble-copilot';
    bubble.innerHTML = html;
    
    // Add copy and insert buttons to code blocks (only for copilot messages)
    if (!isUser) {
      const codeBlocks = bubble.querySelectorAll('pre code');
      codeBlocks.forEach((codeBlock: Element) => {
        const pre = codeBlock.parentElement;
        if (!pre) return;
        
        const toolbar = document.createElement('div');
        toolbar.className = 'fuse-code-toolbar';
        
        const copyBtn = document.createElement('button');
        copyBtn.textContent = '📋 Copy';
        copyBtn.className = 'fuse-code-btn';
        copyBtn.onclick = () => {
          const text = codeBlock.textContent || '';
          navigator.clipboard.writeText(text).then(() => {
            copyBtn.textContent = '✅ Copied!';
            setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
          });
        };
        
        const insertBtn = document.createElement('button');
        insertBtn.textContent = '⬇️ Insert';
        insertBtn.className = 'fuse-code-btn';
        insertBtn.onclick = () => {
          const text = codeBlock.textContent || '';
          if (insertCallback) {
            insertCallback(text);
            insertBtn.textContent = '✅ Inserted!';
            setTimeout(() => { insertBtn.textContent = '⬇️ Insert'; }, 2000);
          }
        };
        
        toolbar.appendChild(copyBtn);
        toolbar.appendChild(insertBtn);
        pre.style.position = 'relative';
        pre.insertBefore(toolbar, pre.firstChild);
      });
    }
    
    const meta = document.createElement('div'); meta.className = 'fuse-chat-meta'; meta.textContent = new Date().toLocaleTimeString();
    const w = document.createElement('div'); w.className = isUser ? 'fuse-chat-msg-user' : 'fuse-chat-msg-copilot';
    w.appendChild(bubble);
    w.appendChild(meta);
    line.appendChild(w);
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  // keyboard support: Enter to send, Shift+Enter newline
  input.addEventListener('keydown', (ev: KeyboardEvent) => {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      send.click();
    }
  });

  clearBtn.addEventListener('click', () => { log.innerHTML = ''; });
  exportBtn.addEventListener('click', () => {
    const data = log.innerText;
    const blob = new Blob([data], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'fuse_chat.txt'; a.click(); URL.revokeObjectURL(url);
  });

  wrapper.appendChild(log);
  wrapper.appendChild(controls);

  (async ()=>{
    try{
      // Load chat-specific styles first
      const chatStyles = await fetch('/fuse/static/chat-styles.css');
      if(chatStyles.ok){
        const css = await chatStyles.text();
        const st = document.createElement('style');
        st.textContent = css;
        document.head.appendChild(st);
      }
    }catch(e){ /* ignore styles */ }
    
    try{
      const s = await fetch('/fuse/static/styles.json');
      if(s.ok){
        const styles = await s.json();
        const css = Object.values(styles).join('\n');
        const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
        wrapper.className = 'fuse-chat-wrapper';
        log.className = 'fuse-chat-log';
        select.className = 'fuse-chat-engine';
        input.className = 'fuse-chat-input';
        send.className = 'fuse-chat-button';
        clearBtn.className = 'fuse-chat-small';
        exportBtn.className = 'fuse-chat-small';
        streamLabel.className = 'fuse-chat-meta';
      }
    }catch(e){ /* ignore styles */ }

    try{
      const r = await fetch('/fuse/api/llm/engines');
      const engines = await r.json();
      if(Array.isArray(engines)){
        engines.forEach((e:any)=>{ const opt = document.createElement('option'); opt.value = String(e); opt.textContent = String(e); select.appendChild(opt); });
      } else if(typeof engines === 'object'){
        Object.entries(engines).forEach(([k,v])=>{ const opt = document.createElement('option'); opt.value = String(k); opt.textContent = String(v); select.appendChild(opt); });
      }
    }catch(e){ const err = document.createElement('div'); err.textContent = 'Failed to load engines: '+String(e); log.appendChild(err);}  
  })();

  let es: any = null; // EventSource
  let isLoading = false;
  
  // Callback to insert code into notebook
  const insertToNotebook = async (text: string) => {
    try {
      // Try to use clipboard as fallback
      await navigator.clipboard.writeText(text);
      // Show toast notification
      const toast = document.createElement('div');
      toast.textContent = '📋 Code copied! Paste into notebook with Cmd/Ctrl+V';
      toast.style.cssText = 'position:fixed;top:20px;right:20px;background:#4caf50;color:white;padding:12px 20px;border-radius:4px;z-index:10000;';
      document.body.appendChild(toast);
      setTimeout(() => toast.remove(), 3000);
    } catch (e) {
      console.error('Insert failed', e);
    }
  };
  
  function setLoading(loading: boolean) {
    isLoading = loading;
    send.disabled = loading;
    input.disabled = loading;
    send.textContent = loading ? '⏳ Thinking...' : 'Send';
    if (loading) {
      const loadingDiv = document.createElement('div');
      loadingDiv.id = 'fuse-chat-loading';
      loadingDiv.className = 'fuse-chat-loading';
      loadingDiv.innerHTML = '💭 Thinking...';
      log.appendChild(loadingDiv);
      log.scrollTop = log.scrollHeight;
    } else {
      const loadingDiv = document.getElementById('fuse-chat-loading');
      if (loadingDiv) loadingDiv.remove();
    }
  }
  
  send.addEventListener('click', async ()=>{
    const text = (input as HTMLTextAreaElement).value;
    if(!text || isLoading) return;
    const engine = (select as HTMLSelectElement).value || 'think';
    const useStream = (streamToggle as HTMLInputElement).checked;
    
    await renderMessage(text, true, insertToNotebook);
    (input as HTMLTextAreaElement).value = '';
    setLoading(true);

    if(useStream){
      if(es){ es.close(); es = null; }
      try{
        es = new EventSource('/fuse/api/llm/stream?engine='+encodeURIComponent(engine));
        es.onmessage = (ev:any)=>{
          try{ 
            const d = JSON.parse(ev.data); 
            if (d.error) {
              renderMessage('❌ ' + d.error + (d.suggestion ? '\n\n💡 ' + d.suggestion : ''), false, insertToNotebook);
              setLoading(false);
              if(es){ es.close(); es=null;}
            } else {
              renderMessage(d?.delta || JSON.stringify(d), false, insertToNotebook); 
            }
          }catch(e){ renderMessage(ev.data, false);}        
        };
        es.onerror = (ev:any)=>{ 
          renderMessage('⚠️ Stream closed or connection error', false, insertToNotebook); 
          setLoading(false);
          if(es){ es.close(); es=null;} 
        };
        await fetch('/fuse/api/llm', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({engine, messages:[{role:'user', content:text}], stream:true})});
        setLoading(false);
      }catch(e){ 
        renderMessage('❌ Stream error: '+String(e), false);
        setLoading(false);
      }    
    } else {
      try{
        const r = await fetch('/fuse/api/llm', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({engine, messages:[{role:'user', content:text}], stream:false})});
        if (!r.ok) {
          const err = await r.json().catch(() => ({error: 'HTTP ' + r.status}));
          throw new Error(err.error + (err.suggestion ? ' (💡 ' + err.suggestion + ')' : ''));
        }
        const j = await r.json();
        const content = j?.choices?.[0]?.message?.content || JSON.stringify(j);
        await renderMessage(content, false, insertToNotebook);
      }catch(e){ 
        await renderMessage('❌ Error: '+String(e), false, insertToNotebook);
      } finally {
        setLoading(false);
      }    
    }
  });

  return wrapper;
}

export function ChatWidget() {
  const ref = React.useRef<HTMLDivElement | null>(null as any);
  React.useEffect(()=>{
    if(ref.current){
      const el = createChatWidgetElement();
      ref.current.appendChild(el);
      return ()=>{ if(ref.current){ ref.current.innerHTML = ''; } };
    }
  }, []);
  return React.createElement('div', { ref });
}
