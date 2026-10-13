"""Dedicated tool routes reuse the native workspace's data and analysis engines."""
import json

TOOLS = {
    'crypto-scanner': ('Crypto address scanner', 'Research', 'Inspect Ethereum activity and direct Tornado Cash interactions.'),
    'screener': ('Stock screener', 'Markets', 'Filter your universe. Find your next opportunity.'),
    'markets': ('Market overview', 'Markets', 'A broader view of your markets, movers and watchlists.'),
    'rotation': ('Sector rotation', 'Markets', 'Compare sector momentum across timeframes.'),
    'fundamentals': ('Company fundamentals', 'Markets', 'Explore the business behind the symbol.'),
    'liquidity': ('Volume & liquidity', 'Markets', 'Understand participation, turnover and trading activity.'),
    'data': ('Price history', 'Markets', 'Inspect stored bars and export the data behind your research.'),
    'strategy': ('Strategy tester', 'Research', 'Test your rules against historical price data.'),
    'compare': ('Strategy comparison', 'Research', 'Compare strategies on the same history and costs.'),
    'pairs': ('Pairs analysis', 'Research', 'Explore relationships, spreads and correlations.'),
    'forecast': ('Forecast lab', 'Research', 'Evaluate forecasts against a no-change baseline.'),
    'markov': ('Markov analysis', 'Research', 'Explore regimes and state transition probabilities.'),
    'brownian': ('Monte Carlo simulations', 'Research', 'Model a range of possible price paths.'),
    'portfolio': ('Portfolio & allocation', 'Portfolio', 'Bring holdings, allocation and stress scenarios together.'),
    'risk': ('Returns & risk', 'Portfolio', 'Understand returns, drawdowns and historical exposure.'),
    'options': ('Options lab', 'Portfolio', 'Explore theoretical pricing, Greeks and payoff scenarios.'),
    'ideas': ('Idea notebook', 'Personal', 'Capture, organize and revisit your market ideas.'),
    'journal': ('Research journal', 'Personal', 'Keep your notes and sources connected to a symbol.'),
}


def render_tool_page(root, key, snapshot=False):
    if key == 'crypto-scanner':
        return (root / 'web' / 'crypto-scanner.html').read_text(encoding='utf-8')
    html = (root / 'web' / 'index.html').read_text(encoding='utf-8-sig')
    title, _, _ = TOOLS[key]
    start, end = html.index('<title>'), html.index('</title>') + len('</title>')
    html = html[:start] + f'<title>{title} | QuantStack</title>' + html[end:]
    html = html.replace('<head>', '<head><base href="/workspace/">', 1)
    html = html.replace('<body>', f'<body data-tool-page="{key}">', 1)
    config = json.dumps(TOOLS)
    html = html.replace('</head>', '<link rel="stylesheet" href="/tool-page.css"></head>', 1)
    html = html.replace('<script>init().then(() => window.openHomeDestination?.());</script>',
                        '<script id="toolRegistry" type="application/json">' + config + '</script>'
                        '<script src="/tool-page.js"></script>'
                        '<script>window.startToolPage();</script>')
    if snapshot:
        html = html.replace('<script src="./app.js">',
                            '<script src="./static-data.js"></script><script src="./app.js">')
    if key == 'screener':
        html = html.replace('</head>', '<link rel="stylesheet" href="/screener-page.css"></head>')
        html = html.replace('<script>window.startToolPage();</script>',
                            '<script src="/screener-page.js"></script><script>window.startToolPage();</script>')
    return html
