import pathlib


def test_copy_lines_have_no_shell_redirection():
    p = pathlib.Path('docker/jupyter/Dockerfile').read_text()
    for line in p.splitlines():
        stripped = line.strip()
        # Only check COPY lines touching jupyter paths
        if stripped.startswith('COPY') and 'jupyter/' in stripped:
            assert '2>/dev/null' not in line, f"Found shell redirect in Docker COPY: {line}"
            assert '|| true' not in line, f"Found shell fallback in Docker COPY: {line}"
