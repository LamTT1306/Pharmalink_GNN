import sqlite3
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash


class Database:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS drugs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    drug_idx INTEGER UNIQUE NOT NULL,
                    drug_db_id TEXT,
                    name TEXT NOT NULL,
                    smiles TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS diseases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    disease_idx INTEGER UNIQUE NOT NULL,
                    omim_code TEXT,
                    name TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS proteins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    protein_idx INTEGER UNIQUE NOT NULL,
                    uniprot_id TEXT,
                    sequence TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS known_associations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    drug_idx INTEGER NOT NULL,
                    disease_idx INTEGER NOT NULL,
                    source TEXT DEFAULT 'dataset',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(drug_idx, disease_idx),
                    FOREIGN KEY (drug_idx) REFERENCES drugs(drug_idx) ON DELETE CASCADE,
                    FOREIGN KEY (disease_idx) REFERENCES diseases(disease_idx) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    query_type TEXT NOT NULL,
                    query_idx INTEGER NOT NULL,
                    query_name TEXT NOT NULL,
                    top_k INTEGER DEFAULT 10,
                    results TEXT NOT NULL,
                    model_used TEXT DEFAULT 'similarity',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
            ''')

    # ─── Users ────────────────────────────────────────────────
    def create_user(self, username, email, password, role='user'):
        try:
            with self.get_connection() as conn:
                conn.execute(
                    'INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)',
                    (username, email, generate_password_hash(password), role)
                )
            return True, None
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return False, 'Tên đăng nhập đã tồn tại'
            return False, 'Email đã tồn tại'

    def get_user_by_username(self, username):
        with self.get_connection() as conn:
            return conn.execute(
                'SELECT * FROM users WHERE username = ? AND is_active = 1', (username,)
            ).fetchone()

    def get_user(self, user_id):
        with self.get_connection() as conn:
            return conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    def verify_password(self, user, password):
        return check_password_hash(user['password_hash'], password)

    def update_last_login(self, user_id):
        with self.get_connection() as conn:
            conn.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))

    def get_all_users(self, page=1, per_page=20):
        with self.get_connection() as conn:
            offset = (page - 1) * per_page
            rows = conn.execute(
                'SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?', (per_page, offset)
            ).fetchall()
            total = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
            return rows, total

    def update_user_role(self, user_id, role):
        with self.get_connection() as conn:
            conn.execute('UPDATE users SET role = ? WHERE id = ?', (role, user_id))

    def toggle_user_active(self, user_id):
        with self.get_connection() as conn:
            conn.execute('UPDATE users SET is_active = 1 - is_active WHERE id = ?', (user_id,))

    def count_users(self):
        with self.get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]

    # ─── Drugs ────────────────────────────────────────────────
    def import_drugs(self, drugs_list):
        with self.get_connection() as conn:
            for drug in drugs_list:
                try:
                    conn.execute(
                        'INSERT OR IGNORE INTO drugs (drug_idx, drug_db_id, name, smiles) VALUES (?, ?, ?, ?)',
                        (drug['idx'], drug.get('id', ''), drug['name'], drug.get('smiles', ''))
                    )
                except Exception:
                    pass

    def get_all_drugs(self, page=1, per_page=50, search=''):
        with self.get_connection() as conn:
            offset = (page - 1) * per_page
            if search:
                rows = conn.execute(
                    'SELECT * FROM drugs WHERE name LIKE ? OR drug_db_id LIKE ? ORDER BY drug_idx LIMIT ? OFFSET ?',
                    (f'%{search}%', f'%{search}%', per_page, offset)
                ).fetchall()
                total = conn.execute(
                    'SELECT COUNT(*) FROM drugs WHERE name LIKE ? OR drug_db_id LIKE ?',
                    (f'%{search}%', f'%{search}%')
                ).fetchone()[0]
            else:
                rows = conn.execute(
                    'SELECT * FROM drugs ORDER BY drug_idx LIMIT ? OFFSET ?', (per_page, offset)
                ).fetchall()
                total = conn.execute('SELECT COUNT(*) FROM drugs').fetchone()[0]
            return rows, total

    def get_drug_by_id(self, drug_id):
        with self.get_connection() as conn:
            return conn.execute('SELECT * FROM drugs WHERE id = ?', (drug_id,)).fetchone()

    def update_drug(self, drug_id, name, smiles, description):
        with self.get_connection() as conn:
            conn.execute(
                'UPDATE drugs SET name = ?, smiles = ?, description = ? WHERE id = ?',
                (name, smiles, description, drug_id)
            )

    def count_drugs(self):
        with self.get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM drugs').fetchone()[0]

    # ─── Diseases ─────────────────────────────────────────────
    def import_diseases(self, diseases_list):
        with self.get_connection() as conn:
            for d in diseases_list:
                try:
                    conn.execute(
                        'INSERT OR IGNORE INTO diseases (disease_idx, omim_code, name) VALUES (?, ?, ?)',
                        (d['idx'], d['code'], d['name'])
                    )
                except Exception:
                    pass

    def get_all_diseases(self, page=1, per_page=50, search=''):
        with self.get_connection() as conn:
            offset = (page - 1) * per_page
            if search:
                rows = conn.execute(
                    'SELECT * FROM diseases WHERE name LIKE ? OR omim_code LIKE ? ORDER BY disease_idx LIMIT ? OFFSET ?',
                    (f'%{search}%', f'%{search}%', per_page, offset)
                ).fetchall()
                total = conn.execute(
                    'SELECT COUNT(*) FROM diseases WHERE name LIKE ? OR omim_code LIKE ?',
                    (f'%{search}%', f'%{search}%')
                ).fetchone()[0]
            else:
                rows = conn.execute(
                    'SELECT * FROM diseases ORDER BY disease_idx LIMIT ? OFFSET ?', (per_page, offset)
                ).fetchall()
                total = conn.execute('SELECT COUNT(*) FROM diseases').fetchone()[0]
            return rows, total

    def get_disease_by_id(self, disease_id):
        with self.get_connection() as conn:
            return conn.execute('SELECT * FROM diseases WHERE id = ?', (disease_id,)).fetchone()

    def update_disease(self, disease_id, name, description):
        with self.get_connection() as conn:
            conn.execute(
                'UPDATE diseases SET name = ?, description = ? WHERE id = ?',
                (name, description, disease_id)
            )

    def count_diseases(self):
        with self.get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM diseases').fetchone()[0]

    # ─── Known Associations ───────────────────────────────────
    def import_associations(self, associations):
        with self.get_connection() as conn:
            for assoc in associations:
                try:
                    conn.execute(
                        'INSERT OR IGNORE INTO known_associations (drug_idx, disease_idx) VALUES (?, ?)',
                        (int(assoc[0]), int(assoc[1]))
                    )
                except Exception:
                    pass

    def get_associations(self, page=1, per_page=50, drug_filter='', disease_filter=''):
        with self.get_connection() as conn:
            offset = (page - 1) * per_page
            base = '''
                SELECT ka.id, ka.drug_idx, ka.disease_idx, ka.source, ka.created_at,
                       dr.name AS drug_name, dr.drug_db_id,
                       di.omim_code, di.name AS disease_name
                FROM known_associations ka
                LEFT JOIN drugs dr ON dr.drug_idx = ka.drug_idx
                LEFT JOIN diseases di ON di.disease_idx = ka.disease_idx
            '''
            where, params = [], []
            if drug_filter:
                where.append('(dr.name LIKE ? OR dr.drug_db_id LIKE ?)')
                params += [f'%{drug_filter}%', f'%{drug_filter}%']
            if disease_filter:
                where.append('(di.name LIKE ? OR di.omim_code LIKE ?)')
                params += [f'%{disease_filter}%', f'%{disease_filter}%']
            cond = (' WHERE ' + ' AND '.join(where)) if where else ''
            rows = conn.execute(
                base + cond + ' ORDER BY ka.id LIMIT ? OFFSET ?',
                params + [per_page, offset]
            ).fetchall()
            total = conn.execute(
                'SELECT COUNT(*) FROM known_associations ka LEFT JOIN drugs dr ON dr.drug_idx=ka.drug_idx LEFT JOIN diseases di ON di.disease_idx=ka.disease_idx' + cond,
                params
            ).fetchone()[0]
            return rows, total

    def count_associations(self):
        with self.get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM known_associations').fetchone()[0]

    # ─── Predictions / History ────────────────────────────────
    def save_prediction(self, user_id, query_type, query_idx, query_name, top_k, results, model_used='similarity'):
        with self.get_connection() as conn:
            conn.execute(
                'INSERT INTO predictions (user_id, query_type, query_idx, query_name, top_k, results, model_used) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (user_id, query_type, int(query_idx), query_name, top_k, json.dumps(results), model_used)
            )

    def get_user_predictions(self, user_id, page=1, per_page=15):
        with self.get_connection() as conn:
            offset = (page - 1) * per_page
            rows = conn.execute(
                'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?',
                (user_id, per_page, offset)
            ).fetchall()
            total = conn.execute(
                'SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,)
            ).fetchone()[0]
            return rows, total

    def count_predictions(self):
        with self.get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]

    def get_statistics(self):
        def _to_dicts(rows):
            return [dict(r) for r in rows]

        with self.get_connection() as conn:
            stats = {}
            stats['predictions_per_day'] = _to_dicts(conn.execute('''
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM predictions
                WHERE created_at >= DATE('now', '-30 days')
                GROUP BY DATE(created_at) ORDER BY date
            ''').fetchall())
            stats['top_drugs'] = _to_dicts(conn.execute('''
                SELECT query_name, COUNT(*) as count FROM predictions
                WHERE query_type='drug' GROUP BY query_name ORDER BY count DESC LIMIT 10
            ''').fetchall())
            stats['top_diseases'] = _to_dicts(conn.execute('''
                SELECT query_name, COUNT(*) as count FROM predictions
                WHERE query_type='disease' GROUP BY query_name ORDER BY count DESC LIMIT 10
            ''').fetchall())
            stats['by_type'] = _to_dicts(conn.execute('''
                SELECT query_type, COUNT(*) as count FROM predictions GROUP BY query_type
            ''').fetchall())
            stats['per_user'] = _to_dicts(conn.execute('''
                SELECT u.username, COUNT(p.id) as count
                FROM users u LEFT JOIN predictions p ON p.user_id = u.id
                GROUP BY u.id ORDER BY count DESC LIMIT 10
            ''').fetchall())
            stats['recent_predictions'] = _to_dicts(conn.execute('''
                SELECT p.id, p.query_type, p.query_name, p.created_at, u.username
                FROM predictions p LEFT JOIN users u ON u.id = p.user_id
                ORDER BY p.created_at DESC LIMIT 10
            ''').fetchall())
            stats['total_predictions'] = conn.execute(
                'SELECT COUNT(*) FROM predictions').fetchone()[0]
            stats['active_users'] = conn.execute(
                'SELECT COUNT(DISTINCT user_id) FROM predictions').fetchone()[0]
            return stats

    def import_proteins(self, proteins_list):
        with self.get_connection() as conn:
            for p in proteins_list:
                try:
                    conn.execute(
                        'INSERT OR IGNORE INTO proteins (protein_idx, uniprot_id, sequence) VALUES (?, ?, ?)',
                        (p['idx'], p.get('id', ''), p.get('sequence', ''))
                    )
                except Exception:
                    pass

    def get_all_proteins(self, page=1, per_page=50, search=''):
        with self.get_connection() as conn:
            offset = (page - 1) * per_page
            if search:
                rows = conn.execute(
                    'SELECT * FROM proteins WHERE uniprot_id LIKE ? OR description LIKE ? ORDER BY protein_idx LIMIT ? OFFSET ?',
                    (f'%{search}%', f'%{search}%', per_page, offset)
                ).fetchall()
                total = conn.execute(
                    'SELECT COUNT(*) FROM proteins WHERE uniprot_id LIKE ? OR description LIKE ?',
                    (f'%{search}%', f'%{search}%')
                ).fetchone()[0]
            else:
                rows = conn.execute(
                    'SELECT * FROM proteins ORDER BY protein_idx LIMIT ? OFFSET ?', (per_page, offset)
                ).fetchall()
                total = conn.execute('SELECT COUNT(*) FROM proteins').fetchone()[0]
            return rows, total

    def count_proteins(self):
        with self.get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM proteins').fetchone()[0]

    def data_imported(self):
        with self.get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM drugs').fetchone()[0] > 0
