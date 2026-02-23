const fs = require('fs');
const path = require('path');
const { workspace, window } = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');

let client;

function activate(context) {
    const resolved = resolveServerCommand(context);
    if (!resolved) {
        window.showInformationMessage('Fuse language server could not be located. Check `fuse.languageServerCommand` or run the extension build script.');
        return;
    }

    const serverOptions = {
        command: resolved.command,
        args: resolved.args,
        transport: TransportKind.stdio,
        options: resolved.options,
    };

    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'fuse' }]
    };

    client = new LanguageClient('fuseLanguageServer', 'Fuse Language Server', serverOptions, clientOptions);
    client.start();
}

function resolveServerCommand(context) {
    const cfg = workspace.getConfiguration('fuse');
    const override = (cfg.get('languageServerCommand') || '').trim();
    if (override) {
        const parts = override.split(/\s+/);
        return { command: parts[0], args: parts.slice(1), options: {} };
    }

    const binary = process.platform === 'win32' ? 'fuse-lsp.exe' : 'fuse-lsp';
    const bundled = context.asAbsolutePath(path.join('server', binary));
    if (fs.existsSync(bundled)) {
        return { command: bundled, args: [], options: {} };
    }

    const python = process.env.FUSE_SERVER_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
    const env = { ...process.env, PYTHONPATH: context.extensionPath };
    return { command: python, args: ['-m', 'src.lsp_server'], options: { env } };
}

function deactivate() {
    if (!client) {
        return undefined;
    }
    return client.stop();
}

module.exports = { activate, deactivate };
