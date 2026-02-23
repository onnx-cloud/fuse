import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ICompletionManager } from '@jupyterlab/completer';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { ErrorCard } from './errorCard';

const extension: JupyterFrontEndPlugin<void> = {
  id: '@fuse/jupyterlab-fuse',
  autoStart: true,
  requires: [ICompletionManager],
  activate: (app: JupyterFrontEnd, completionManager: ICompletionManager) => {
    console.log('Fuse frontend extension loaded');

    // Register a simple command to fetch ops and show in console
    app.commands.addCommand('fuse:show-ops', {
      label: 'Show Fuse Ops',
      execute: async () => {
        try {
          const r = await fetch('/fuse/api/ops');
          const ops = await r.json();
          // For now, just print to console
          console.log('Fuse ops:', ops.slice(0, 50));
        } catch (e) {
          console.error('Failed to fetch ops', e);
        }
      },
    });

    // Add a command to open a welcome widget (iframe to /fuse/welcome)
    app.commands.addCommand('fuse:open-welcome-widget', {
      label: 'Open Fuse Welcome',
      execute: () => {
        const { Widget } = require('@lumino/widgets');
        const iframe = document.createElement('iframe');
        iframe.style.border = 'none';
        iframe.style.width = '100%';
        iframe.style.height = '100%';
        iframe.src = '/fuse/welcome';
        const w = new Widget({ node: iframe });
        w.id = 'fuse-welcome-widget';
        w.title.label = 'Fuse';
        w.title.closable = true;
        app.shell.add(w, 'main');
      },
    });

    // Add a command to open a Copilot Chat widget
    app.commands.addCommand('fuse:open-chat', {
      label: 'Open Copilot Chat',
      execute: () => {
        const { Widget } = require('@lumino/widgets');
        const container = document.createElement('div');
        container.style.height = '100%';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        const w = new Widget({ node: container });
        w.id = 'fuse-copilot-widget';
        w.title.label = 'Fuse Copilot';
        w.title.closable = true;
        app.shell.add(w, 'main');

        // Lazy-load and render the chat UI
        import('./chatWidget').then(mod => {
          const root = (window as any).ReactDOM.createRoot(container);
          const el = mod.createChatWidgetElement();
          // expose a small API to insert to active notebook via app.commands if available
          const actions = {
            insertToNotebook: async (text: string) => {
              try {
                await app.commands.execute('notebook:insert-cell-below');
                // client must paste content - we fallback to clipboard
                await navigator.clipboard.writeText(text);
              } catch (e) {
                await navigator.clipboard.writeText(text);
              }
            }
          };
          root.render(mod.ChatWidget());
          // also append our element inside the rendered container for robust non-React behavior
          container.appendChild(el);
        }).catch(e => console.error('failed to load chat widget', e));
      },
    });

    app.commands.addCommand('fuse:manage-llm', {
      label: 'Manage LLM Engines',
      execute: () => {
        const { Widget } = require('@lumino/widgets');
        const container = document.createElement('div');
        container.style.height = '100%';
        container.style.padding = '12px';
        const w = new Widget({ node: container });
        w.id = 'fuse-llm-admin';
        w.title.label = 'LLM Engines';
        w.title.closable = true;
        app.shell.add(w, 'main');
        import('./adminWidget').then(mod => {
          const root = (window as any).ReactDOM.createRoot(container);
          root.render(mod.AdminWidget());
        }).catch(e => {
          console.error('failed to load admin widget', e);
          container.textContent = 'Failed to load admin UI: ' + String(e);
        });
      }
    });
    // Add a command to open an Error Card widget (demonstration uses a static payload)
    app.commands.addCommand('fuse:show-error-card', {
      label: 'Show Fuse Error',
      execute: async () => {
        try {
          // For demo, call map_error with a sample message; a real integration would POST real error data
          const r = await fetch('/fuse/api/map_error', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: 'Demo error from frontend' }),
          });
          const err = await r.json();

          const { Widget } = require('@lumino/widgets');
          const container = document.createElement('div');
          container.style.padding = '12px';
          const root = ReactDOM.createRoot(container);
          root.render(React.createElement(ErrorCard, { error: err }));

          const w = new Widget({ node: container });
          w.id = 'fuse-error-widget';
          w.title.label = 'Fuse Error';
          w.title.closable = true;
          app.shell.add(w, 'main');
        } catch (e) {
          console.error('Failed to show error card', e);
        }
      },
    });

    // Add a launcher item for the welcome widget if launcher is present
    try {
      const { ILauncher } = require('@jupyterlab/launcher');
      // If a launcher is available, add an item
    } catch (_) {
      // ignore; launcher may not be present in some contexts
    }

    // Context-aware completion provider
    completionManager.register({
      identifier: 'fuse-autocomplete',
      renderer: null as any,
      fetch: async (request: any) => {
        // request has 'text' (current line) and 'position' (cursor offset)
        try {
          const text = request.text || '';
          const pos = request.position || 0;
          
          // Extract prefix (word before cursor)
          const beforeCursor = text.slice(0, pos);
          const match = beforeCursor.match(/[\w]+$/);
          const prefix = match ? match[0] : '';
          
          // Get context (30 chars before cursor for context detection)
          const context = text.slice(Math.max(0, pos - 30), pos);
          
          const r = await fetch('/fuse/api/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prefix, context }),
          });
          const items = await r.json();
          
          // Map to JupyterLab CompletionItem format
          return items.map((it: any) => ({
            label: it.label,
            insertText: it.insertText,
            type: it.kind || 'function',
            documentation: it.detail
          }));
        } catch (e) {
          console.error('completion error', e);
          return [];
        }
      },
    } as any);

    // Register keyboard shortcuts
    app.commands.addKeyBinding({
      command: 'fuse:open-chat',
      keys: ['Accel K'],  // Cmd+K on Mac, Ctrl+K on Windows/Linux
      selector: 'body',
    });

    app.commands.addKeyBinding({
      command: 'fuse:open-welcome-widget',
      keys: ['Accel Shift H'],  // Cmd+Shift+H / Ctrl+Shift+H for Help/Welcome
      selector: 'body',
    });

    console.log('Fuse keyboard shortcuts registered: Cmd/Ctrl+K (Chat), Cmd/Ctrl+Shift+H (Welcome)');
  },
};

export default extension;
