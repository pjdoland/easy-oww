#!/bin/bash
# Fix entry point script for editable install
SCRIPT_PATH="venv/bin/easy-oww"
if [ -f "$SCRIPT_PATH" ]; then
    cat > "$SCRIPT_PATH" << 'ENTRY'
#!/Users/pjdoland/Desktop/Repos/easy-oww/venv/bin/python3.13
# -*- coding: utf-8 -*-
import re
import sys
import os

# Add the project root to sys.path for editable install
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from easy_oww.cli.main import cli

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cli())
ENTRY
    chmod +x "$SCRIPT_PATH"
    echo "Fixed easy-oww entry point"
fi
