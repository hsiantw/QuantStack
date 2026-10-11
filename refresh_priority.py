"""Put due priority assets first while retaining oldest-attempt rotation."""
from datetime import datetime, timedelta, timezone
import json


def order_symbols(symbols, attempts, data, freshness_minutes):
    symbols = list(dict.fromkeys(symbols))
    try:
        priority = json.loads((data / 'refresh-priority.json').read_text(encoding='utf-8'))['symbols']
    except (OSError, ValueError, KeyError):
        priority = []
    # Bound the fast lane so the rest of the universe keeps rotating.
    rank = {symbol: index for index, symbol in enumerate(priority[:120])}
    now = datetime.now(timezone.utc)
    def due(symbol):
        try:
            last = datetime.fromisoformat(attempts[symbol])
            age = now - last
            return age >= timedelta(minutes=freshness_minutes) or age < timedelta(0)
        except (KeyError, ValueError, TypeError):
            return True
    return sorted(symbols, key=lambda s: (0, rank[s]) if s in rank and due(s)
                  else (1, attempts.get(s, ''), s))
