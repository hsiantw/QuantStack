"""Account sessions and versioned personal workspaces for the chart gateway."""
import asyncio
from contextlib import closing
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from urllib.parse import urlsplit

from aiohttp import web

COOKIE = 'quantstack_session'
TTL = 7 * 24 * 3600
LIMIT = 2_000_000


class AccountError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=32768, r=8, p=3,
                          maxmem=64 * 1024 * 1024).hex()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Accounts:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                    salt BLOB NOT NULL, password_hash TEXT NOT NULL,
                    workspace TEXT NOT NULL DEFAULT '{}', revision INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER);
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL, expires INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS auth_limits (
                    key TEXT PRIMARY KEY, started INTEGER NOT NULL, count INTEGER NOT NULL);
            ''')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def throttle(self, peer, username):
        now = int(time.time())
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM auth_limits WHERE started < ?', (now - 900,))
            for key, maximum in [('ip:' + digest(peer), 30), ('user:' + digest(username), 15)]:
                row = db.execute('SELECT count FROM auth_limits WHERE key=?', (key,)).fetchone()
                if row and row['count'] >= maximum:
                    raise AccountError(429, 'Too many sign-in attempts. Try again in 15 minutes.')
                db.execute('INSERT INTO auth_limits VALUES (?, ?, 1) ON CONFLICT(key) DO UPDATE SET count=count+1', (key, now))

    def authenticate(self, payload, peer, register=False, previous=''):
        username, password = payload.get('username'), payload.get('password')
        if not isinstance(username, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{3,32}', username):
            raise AccountError(400, 'Use a username of 3–32 letters, numbers, dots, underscores or hyphens.')
        if not isinstance(password, str) or not 12 <= len(password) <= 128:
            raise AccountError(400, 'Use a password of 12–128 characters.')
        username = username.lower()
        self.throttle(peer, username)
        with closing(self.connect()) as db, db:
            if register:
                salt = secrets.token_bytes(16)
                hashed = password_hash(password, salt)
                try:
                    db.execute('INSERT INTO accounts(id,username,salt,password_hash) VALUES (?,?,?,?)',
                               (secrets.token_hex(16), username, salt, hashed))
                except sqlite3.IntegrityError:
                    raise AccountError(409, 'That username is unavailable.') from None
            user = db.execute('SELECT * FROM accounts WHERE username=?', (username,)).fetchone()
            # Perform equivalent work for unknown usernames.
            if not register:
                candidate = password_hash(password, user['salt'] if user else b'\0' * 16)
                if not user or not hmac.compare_digest(candidate, user['password_hash']):
                    raise AccountError(401, 'Invalid username or password.')
            token = secrets.token_urlsafe(32)
            db.execute('DELETE FROM sessions WHERE expires <= ? OR token_hash=?', (int(time.time()), digest(previous)))
            db.execute('INSERT INTO sessions VALUES (?,?,?)', (digest(token), user['id'], int(time.time()) + TTL))
            return self.public(user), token

    @staticmethod
    def public(user):
        return {key: user[key] for key in ('id', 'username', 'revision', 'updated_at')}

    def user(self, db, token):
        user = db.execute('''SELECT a.* FROM accounts a JOIN sessions s ON a.id=s.account_id
                           WHERE s.token_hash=? AND s.expires>?''', (digest(token), int(time.time()))).fetchone()
        if not user:
            raise AccountError(401, 'Sign in to access your saved workspace.')
        return user

    def workspace(self, token, payload=None):
        with closing(self.connect()) as db, db:
            if payload is not None:
                entries, revision = payload.get('entries'), payload.get('revision')
                if (not isinstance(entries, dict) or len(entries) > 3000 or type(revision) is not int or
                    any(not re.fullmatch(r'atlas\.[\w.^= -]{1,180}', key) or not isinstance(value, str)
                        for key, value in entries.items())):
                    raise AccountError(400, 'Invalid workspace data.')
                encoded = json.dumps(entries, ensure_ascii=True)
                if len(encoded.encode()) > LIMIT:
                    raise AccountError(413, 'Your workspace exceeds the 2 MB account storage limit.')
                db.execute('BEGIN IMMEDIATE')
            user = self.user(db, token)
            if payload is not None:
                if payload.get('account_id') != user['id']:
                    raise AccountError(409, 'The signed-in account changed. Reload before saving.')
                if user['revision'] != revision:
                    raise AccountError(409, 'A newer workspace was saved elsewhere. Download a local backup, then load the account workspace before saving again.')
                db.execute('UPDATE accounts SET workspace=?,revision=revision+1,updated_at=? WHERE id=?',
                           (encoded, int(time.time()), user['id']))
                user = self.user(db, token)
            return {'user': self.public(user), 'entries': json.loads(user['workspace'])}

    def session(self, token):
        with closing(self.connect()) as db:
            try:
                return {'user': self.public(self.user(db, token))}
            except AccountError:
                return {'user': None}

    def logout(self, token, account_id):
        with closing(self.connect()) as db, db:
            if self.user(db, token)['id'] != account_id:
                raise AccountError(409, 'The signed-in account changed. Reload before signing out.')
            db.execute('DELETE FROM sessions WHERE token_hash=?', (digest(token),))
        return {'user': None}


def install_accounts(app, path=None, public_origin=None):
    path = path or os.environ.get('QUANTSTACK_ACCOUNTS_DB') or Path(__file__).parent / 'data' / 'accounts.sqlite'
    public_origin = public_origin or os.environ.get('QUANTSTACK_PUBLIC_ORIGIN', '')
    if public_origin:
        parsed = urlsplit(public_origin)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
            raise ValueError('QUANTSTACK_PUBLIC_ORIGIN must be an HTTPS origin without a trailing slash.')
    store = Accounts(path)
    workers = asyncio.Semaphore(4)

    async def endpoint(request):
        headers = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}
        try:
            origin = public_origin or f'{request.scheme}://{request.host}'
            local = request.remote in ('127.0.0.1', '::1') and request.url.host in ('127.0.0.1', 'localhost', '::1')
            if not public_origin and not request.secure and not local:
                raise AccountError(403, 'Account access requires HTTPS outside this computer.')
            if request.method == 'POST' and (request.headers.get('Origin') != origin or request.content_type != 'application/json'):
                raise AccountError(403, 'Use the account controls from this QuantStack site.')
            payload = {}
            if request.method == 'POST':
                chunks, size = [], 0
                async for chunk in request.content.iter_chunked(65536):
                    size += len(chunk)
                    if size > LIMIT:
                        raise AccountError(413, 'Account request exceeds the 2 MB limit.')
                    chunks.append(chunk)
                try:
                    payload = json.loads(b''.join(chunks))
                    if not isinstance(payload, dict):
                        raise ValueError()
                except (ValueError, UnicodeError):
                    raise AccountError(400, 'Expected a JSON object.') from None
            action = request.match_info['action']
            token = request.cookies.get(COOKIE, '')
            new_token = None
            async with workers:
                if action in ('login', 'register') and request.method == 'POST':
                    user, new_token = await asyncio.to_thread(store.authenticate, payload, request.remote or '', action == 'register', token)
                    result = {'user': user}
                elif action == 'logout' and request.method == 'POST':
                    result = await asyncio.to_thread(store.logout, token, payload.get('account_id'))
                elif action == 'session' and request.method == 'GET':
                    result = await asyncio.to_thread(store.session, token)
                elif action == 'workspace' and request.method in ('GET', 'POST'):
                    result = await asyncio.to_thread(store.workspace, token, payload if request.method == 'POST' else None)
                else:
                    raise AccountError(404, 'Account endpoint not found.')
            response = web.json_response(result, headers=headers)
            if new_token:
                response.set_cookie(COOKIE, new_token, max_age=TTL, httponly=True,
                                    secure=bool(public_origin) or request.secure, samesite='Strict', path='/workspace/api/account')
            if action == 'logout':
                response.del_cookie(COOKIE, path='/workspace/api/account')
            return response
        except AccountError as error:
            return web.json_response({'error': error.message}, status=error.status, headers=headers)

    app.router.add_route('*', '/workspace/api/account/{action}', endpoint)
