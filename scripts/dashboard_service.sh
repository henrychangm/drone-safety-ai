#!/bin/bash
# Deja el dashboard siempre encendido (agente de macOS: arranca al iniciar sesión y se reinicia si se cae).
#   scripts/dashboard_service.sh install | uninstall | status | restart | logs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.dronesafety.dashboard"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/drone-safety-dashboard.log"
DOMAIN="gui/$(id -u)"
URL="http://127.0.0.1:8000"

case "${1:-}" in
  install)
    [ -x "$ROOT/.venv/bin/python" ] || { echo "No existe $ROOT/.venv (crea el entorno primero)"; exit 1; }
    mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>-m</string><string>src.dashboard.app</string></array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
PL
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "Servicio instalado. Dashboard: $URL  (logs: $LOG)" ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"; echo "Servicio desinstalado." ;;
  restart)
    launchctl kickstart -k "$DOMAIN/$LABEL"; echo "Reiniciado." ;;
  status)
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state =|pid =|last exit" || echo "Servicio no instalado"
    curl -s -m 3 -o /dev/null -w "HTTP $URL -> %{http_code}\n" "$URL/api/stats" || echo "El dashboard no responde" ;;
  logs)
    tail -n 50 -f "$LOG" ;;
  *) echo "Uso: $0 install | uninstall | status | restart | logs"; exit 1 ;;
esac
