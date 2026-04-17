"""
AMDGT Drug-Disease Association Web Application
Flask Backend
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import json
import secrets
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, g)
from database import Database
from predict import PredictionEngine
import gemini_client
import replicate_client

# ─── App Setup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
# ── Persistent secret key (survives restarts so sessions stay valid) ──
_key_file = os.path.join(BASE_DIR, '.secret_key')
if os.path.exists(_key_file):
    with open(_key_file, 'r') as _f:
        _stored_key = _f.read().strip()
else:
    _stored_key = secrets.token_hex(32)
    with open(_key_file, 'w') as _f:
        _f.write(_stored_key)
app.secret_key = os.environ.get('SECRET_KEY', _stored_key)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 7   # 7 days

DB_PATH = os.path.join(BASE_DIR, 'PharmaLink_GNN.db')
db = Database(DB_PATH)

# ─── Lazy multi-dataset engine ────────────────────────────────
AVAILABLE_DATASETS = {
    'B-dataset': 'B-dataset (269 thuốc, 598 bệnh – tên bệnh tiếng Anh)',
    'C-dataset': 'C-dataset (663 thuốc, 409 bệnh – mã OMIM)',
    'F-dataset': 'F-dataset (593 thuốc, 313 bệnh – mã OMIM)',
}
_engines: dict = {}


def _load_engine(dataset: str) -> 'PredictionEngine':
    if dataset not in _engines:
        _engines[dataset] = PredictionEngine(dataset=dataset)
    return _engines[dataset]


class _EngineProxy:
    """Transparently delegates to the session-selected engine."""
    def __getattr__(self, name):
        try:
            ds = session.get('dataset', 'C-dataset')
        except RuntimeError:
            ds = 'C-dataset'
        return getattr(_load_engine(ds), name)


engine = _EngineProxy()
_load_engine('C-dataset')  # pre-load default at startup

# Auto-configure Gemini if env var is set
_gemini_key_env = os.environ.get('GEMINI_API_KEY', '')
if _gemini_key_env:
    try:
        gemini_client.configure(_gemini_key_env)
        print('  ✓ Gemini AI đã sẵn sàng')
    except Exception as _e:
        print(f'  ⚠ Gemini không khởi động được: {_e}')

# Auto-configure Replicate if env var is set
_replicate_key_env = os.environ.get('REPLICATE_API_TOKEN', '')
if _replicate_key_env:
    try:
        replicate_client.configure(_replicate_key_env)
        print('  ✓ Replicate AI đã sẵn sàng')
    except Exception as _e:
        print(f'  ⚠ Replicate không khởi động được: {_e}')


# ─── Auth Decorators ──────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vui lòng đăng nhập để tiếp tục.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vui lòng đăng nhập.', 'warning')
            return redirect(url_for('login'))
        user = db.get_user(session['user_id'])
        if not user or user['role'] != 'admin':
            flash('Bạn không có quyền truy cập trang này.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')
    g.user = db.get_user(user_id) if user_id else None


# ─── Auth Routes ──────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = db.get_user_by_username(username)
        if user and db.verify_password(user, password):
            session.clear()
            session.permanent = True
            session['user_id'] = user['id']
            session['role'] = user['role']
            db.update_last_login(user['id'])
            flash(f'Chào mừng, {user["username"]}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Tên đăng nhập hoặc mật khẩu không đúng.', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if g.user:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not username or not email or not password:
            flash('Vui lòng điền đầy đủ thông tin.', 'danger')
        elif len(password) < 6:
            flash('Mật khẩu phải có ít nhất 6 ký tự.', 'danger')
        elif password != confirm:
            flash('Mật khẩu xác nhận không khớp.', 'danger')
        else:
            # First user gets admin role
            role = 'admin' if db.count_users() == 0 else 'user'
            ok, err = db.create_user(username, email, password, role)
            if ok:
                flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
                return redirect(url_for('login'))
            flash(err, 'danger')
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Đã đăng xuất.', 'info')
    return redirect(url_for('login'))


# ─── User Routes ──────────────────────────────────────────────
@app.route('/')
@login_required
def dashboard():
    info = engine.get_dataset_info()
    stats = {
        'n_drugs':        info['n_drugs'],
        'n_diseases':     info['n_diseases'],
        'n_proteins':     info['n_proteins'],
        'n_associations': info['n_associations'],
        'n_drpr':         info['n_drpr'],
        'n_dipr':         info['n_dipr'],
        'n_predictions':  db.count_predictions(),
        'models_available': info['models_available'],
        'gnn_ready':      info['gnn_ready'],
        'gnn_auc':        info['gnn_auc'],
    }
    recent, _ = db.get_user_predictions(session['user_id'], page=1, per_page=5)
    return render_template('dashboard.html', stats=stats, recent=recent)


@app.route('/predict')
@login_required
def predict_page():
    return render_template('predict.html')


@app.route('/history')
@login_required
def history():
    page = int(request.args.get('page', 1))
    rows, total = db.get_user_predictions(session['user_id'], page=page, per_page=15)
    total_pages = (total + 14) // 15
    return render_template('history.html', rows=rows, page=page,
                           total=total, total_pages=total_pages)


# ─── User API ─────────────────────────────────────────────────
@app.route('/api/drugs/search')
@login_required
def api_drugs_search():
    q = request.args.get('q', '').strip()
    if len(q) < 1:
        return jsonify([])
    results = engine.search_drugs(q, limit=15)
    return jsonify(results)


@app.route('/api/diseases/search')
@login_required
def api_diseases_search():
    q = request.args.get('q', '').strip()
    if len(q) < 1:
        return jsonify([])
    results = engine.search_diseases(q, limit=15)
    return jsonify(results)


@app.route('/api/proteins/search')
@login_required
def api_proteins_search():
    q = request.args.get('q', '').strip()
    if len(q) < 1:
        return jsonify([])
    results = engine.search_proteins(q, limit=15)
    return jsonify(results)


@app.route('/api/dataset', methods=['GET', 'POST'])
@login_required
def api_dataset():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        ds = data.get('dataset', '')
        if ds not in AVAILABLE_DATASETS:
            return jsonify({'error': 'Dataset không hợp lệ'}), 400
        session['dataset'] = ds
        eng = _load_engine(ds)
        info = eng.get_dataset_info()
        return jsonify({
            'ok': True, 'dataset': ds, 'label': AVAILABLE_DATASETS[ds],
            'n_drugs': info['n_drugs'], 'n_diseases': info['n_diseases'],
            'n_proteins': info['n_proteins'],
        })
    current = session.get('dataset', 'C-dataset')
    return jsonify({
        'dataset': current,
        'available': [
            {'id': k, 'label': v} for k, v in AVAILABLE_DATASETS.items()
        ]
    })


@app.route('/api/models')
def api_models():
    return jsonify(engine.available_models())


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data = request.get_json(silent=True) or {}
    query_type = data.get('query_type', '')
    query_idx  = data.get('query_idx')
    top_k      = min(int(data.get('top_k', 10)), 50)
    model      = data.get('model', 'fuzzy')
    sub_type   = data.get('sub_type', '')   # 'protein' for drug/disease→protein

    if query_type not in ('drug', 'disease', 'protein') or query_idx is None:
        return jsonify({'error': 'Thiếu tham số'}), 400

    try:
        query_idx = int(query_idx)
    except (TypeError, ValueError):
        return jsonify({'error': 'query_idx không hợp lệ'}), 400

    results    = []
    query_name = ''

    if query_type == 'drug':
        info = engine.get_drug_info(query_idx)
        if not info:
            return jsonify({'error': 'Không tìm thấy thuốc'}), 404
        query_name = info['name']
        if sub_type == 'protein':
            results = engine.predict_from_drug_to_proteins(query_idx, top_k=top_k)
        else:
            results = engine.predict_from_drug(query_idx, top_k=top_k, model=model)

    elif query_type == 'disease':
        info = engine.get_disease_info(query_idx)
        if not info:
            return jsonify({'error': 'Không tìm thấy bệnh'}), 404
        query_name = info['name']
        if sub_type == 'protein':
            results = engine.predict_from_disease_to_proteins(query_idx, top_k=top_k)
        else:
            results = engine.predict_from_disease(query_idx, top_k=top_k, model=model)

    else:  # protein
        info = engine.get_protein_info(query_idx)
        if not info:
            return jsonify({'error': 'Không tìm thấy protein'}), 404
        query_name = info['id']
        if sub_type == 'disease':
            results = engine.predict_from_protein_to_diseases(query_idx, top_k=top_k)
        else:
            results = engine.predict_from_protein_to_drugs(query_idx, top_k=top_k)

    model_used = model if query_type in ('drug', 'disease') and sub_type != 'protein' else 'similarity'
    db.save_prediction(
        user_id=session['user_id'], query_type=query_type,
        query_idx=query_idx, query_name=query_name,
        top_k=top_k, results=results, model_used=model_used
    )
    return jsonify({
        'results': results, 'query_name': query_name,
        'query_type': query_type, 'sub_type': sub_type, 'model': model_used
    })


@app.route('/api/predict/matrix', methods=['POST'])
@login_required
def api_predict_matrix():
    data = request.get_json(silent=True) or {}
    drug_idxs    = data.get('drug_idxs', [])
    disease_idxs = data.get('disease_idxs', [])
    model        = data.get('model', 'fuzzy')
    if not drug_idxs or not disease_idxs:
        return jsonify({'error': 'Cần ít nhất 1 thuốc và 1 bệnh'}), 400
    if len(drug_idxs) > 10 or len(disease_idxs) > 10:
        return jsonify({'error': 'Tối đa 10 thuốc và 10 bệnh'}), 400
    try:
        drug_idxs    = [int(i) for i in drug_idxs]
        disease_idxs = [int(i) for i in disease_idxs]
    except (ValueError, TypeError):
        return jsonify({'error': 'Index không hợp lệ'}), 400
    result = engine.predict_matrix(drug_idxs, disease_idxs, model)
    return jsonify(result)


@app.route('/api/fuzzy/explain', methods=['POST'])
@login_required
def api_fuzzy_explain():
    data = request.get_json(silent=True) or {}
    drug_idx    = data.get('drug_idx')
    disease_idx = data.get('disease_idx')
    if drug_idx is None or disease_idx is None:
        return jsonify({'error': 'Thiếu tham số'}), 400
    result = engine.fuzzy_explain(int(drug_idx), int(disease_idx))
    if not result:
        return jsonify({'error': 'Index không hợp lệ'}), 404
    return jsonify(result)


@app.route('/api/molecule/generate', methods=['POST'])
@login_required
def api_molecule_generate():
    data        = request.get_json(silent=True) or {}
    disease_idx = data.get('disease_idx')
    n           = min(int(data.get('n', 6)), 10)
    if disease_idx is None:
        return jsonify({'error': 'Thiếu disease_idx'}), 400
    candidates = engine.generate_candidates(int(disease_idx), n)
    disease    = engine.get_disease_info(int(disease_idx))
    return jsonify({'disease': disease, 'candidates': candidates})


@app.route('/api/graph/entity', methods=['POST'])
@login_required
def api_graph_entity():
    data        = request.get_json(silent=True) or {}
    entity_type = data.get('entity_type', 'drug')
    entity_idx  = data.get('entity_idx')
    top_k       = min(int(data.get('top_k', 12)), 30)
    model       = data.get('model', 'fuzzy')
    if entity_idx is None:
        return jsonify({'error': 'Thiếu entity_idx'}), 400
    result = engine.get_entity_graph(entity_type, int(entity_idx), top_k, model)
    if not result:
        return jsonify({'error': 'Không tìm thấy'}), 404
    return jsonify(result)


@app.route('/api/history/<int:pred_id>')
@login_required
def api_history_detail(pred_id):
    with db.get_connection() as conn:
        row = conn.execute(
            'SELECT * FROM predictions WHERE id = ? AND user_id = ?',
            (pred_id, session['user_id'])
        ).fetchone()
    if not row:
        return jsonify({'error': 'Không tìm thấy'}), 404
    return jsonify({
        'id': row['id'],
        'query_type': row['query_type'],
        'query_name': row['query_name'],
        'top_k': row['top_k'],
        'results': json.loads(row['results']),
        'model_used': row['model_used'],
        'created_at': row['created_at'],
    })


# ─── Admin Routes ─────────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin_dashboard():
    stats = {
        'n_users':        db.count_users(),
        'n_drugs':        db.count_drugs(),
        'n_diseases':     db.count_diseases(),
        'n_proteins':     db.count_proteins(),
        'n_associations': db.count_associations(),
        'n_predictions':  db.count_predictions(),
    }
    stat_data = db.get_statistics()
    return render_template('admin/dashboard.html', stats=stats, stat_data=stat_data)


@app.route('/admin/drugs')
@admin_required
def admin_drugs():
    page = int(request.args.get('page', 1))
    search = request.args.get('q', '').strip()
    rows, total = db.get_all_drugs(page=page, per_page=50, search=search)
    total_pages = (total + 49) // 50
    return render_template('admin/drugs.html', rows=rows, page=page,
                           total=total, total_pages=total_pages, search=search)


@app.route('/admin/drugs/<int:drug_id>/edit', methods=['POST'])
@admin_required
def admin_drug_edit(drug_id):
    name = request.form.get('name', '').strip()
    smiles = request.form.get('smiles', '').strip()
    description = request.form.get('description', '').strip()
    if name:
        db.update_drug(drug_id, name, smiles, description)
        flash('Đã cập nhật thông tin thuốc.', 'success')
    return redirect(url_for('admin_drugs'))


@app.route('/admin/diseases')
@admin_required
def admin_diseases():
    page = int(request.args.get('page', 1))
    search = request.args.get('q', '').strip()
    rows, total = db.get_all_diseases(page=page, per_page=50, search=search)
    total_pages = (total + 49) // 50
    return render_template('admin/diseases.html', rows=rows, page=page,
                           total=total, total_pages=total_pages, search=search)


@app.route('/admin/diseases/<int:disease_id>/edit', methods=['POST'])
@admin_required
def admin_disease_edit(disease_id):
    name = request.form.get('name', '').strip()
    description = request.form.get('description', '').strip()
    if name:
        db.update_disease(disease_id, name, description)
        flash('Đã cập nhật thông tin bệnh.', 'success')
    return redirect(url_for('admin_diseases'))


@app.route('/admin/associations')
@admin_required
def admin_associations():
    page = int(request.args.get('page', 1))
    drug_filter = request.args.get('drug', '').strip()
    disease_filter = request.args.get('disease', '').strip()
    rows, total = db.get_associations(page=page, per_page=50,
                                      drug_filter=drug_filter,
                                      disease_filter=disease_filter)
    total_pages = (total + 49) // 50
    return render_template('admin/associations.html', rows=rows, page=page,
                           total=total, total_pages=total_pages,
                           drug_filter=drug_filter, disease_filter=disease_filter)


@app.route('/admin/proteins')
@admin_required
def admin_proteins():
    page = int(request.args.get('page', 1))
    search = request.args.get('q', '').strip()
    rows, total = db.get_all_proteins(page=page, per_page=50, search=search)
    total_pages = (total + 49) // 50
    return render_template('admin/proteins.html', rows=rows, page=page,
                           total=total, total_pages=total_pages, search=search)


@app.route('/admin/statistics')
@admin_required
def admin_statistics():
    stat_data = db.get_statistics()
    return render_template('admin/statistics.html', stat_data=stat_data)


@app.route('/admin/users')
@admin_required
def admin_users():
    page = int(request.args.get('page', 1))
    rows, total = db.get_all_users(page=page, per_page=20)
    total_pages = (total + 19) // 20
    return render_template('admin/users.html', rows=rows, page=page,
                           total=total, total_pages=total_pages)


@app.route('/admin/users/<int:user_id>/role', methods=['POST'])
@admin_required
def admin_user_role(user_id):
    if user_id == session['user_id']:
        flash('Không thể thay đổi quyền của chính mình.', 'warning')
    else:
        role = request.form.get('role', 'user')
        if role in ('user', 'admin'):
            db.update_user_role(user_id, role)
            flash('Đã cập nhật quyền người dùng.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/toggle', methods=['POST'])
@admin_required
def admin_user_toggle(user_id):
    if user_id == session['user_id']:
        flash('Không thể vô hiệu hoá tài khoản của chính mình.', 'warning')
    else:
        db.toggle_user_active(user_id)
        flash('Đã cập nhật trạng thái người dùng.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/api/stats')
@admin_required
def admin_api_stats():
    stat_data = db.get_statistics()
    return jsonify({
        'predictions_per_day': [
            {'date': r['date'], 'count': r['count']} for r in stat_data['predictions_per_day']
        ],
        'top_drugs': [
            {'name': r['query_name'], 'count': r['count']} for r in stat_data['top_drugs']
        ],
        'top_diseases': [
            {'name': r['query_name'], 'count': r['count']} for r in stat_data['top_diseases']
        ],
        'by_type': [
            {'type': r['query_type'], 'count': r['count']} for r in stat_data['by_type']
        ],
    })


# ─── Gemini AI Routes ───────────────────────────────────────
@app.route('/api/ai/configure', methods=['POST'])
@login_required
def api_ai_configure():
    data = request.get_json(silent=True) or {}
    key = data.get('api_key', '').strip()
    if not key:
        return jsonify({'error': 'Thiếu API key'}), 400
    try:
        gemini_client.configure(key)
        # Persist in session (not stored permanently for security)
        session['gemini_key'] = key
        return jsonify({'ok': True, 'ready': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/replicate/configure', methods=['POST'])
@login_required
def api_replicate_configure():
    data  = request.get_json(silent=True) or {}
    token = data.get('api_token', '').strip()
    if not token:
        return jsonify({'error': 'Thiếu Replicate API token'}), 400
    try:
        replicate_client.configure(token)
        session['replicate_token'] = token
        return jsonify({'ok': True, 'ready': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/replicate/status')
@login_required
def api_replicate_status():
    if not replicate_client.is_ready():
        tok = session.get('replicate_token', '')
        if tok:
            try: replicate_client.configure(tok)
            except Exception: pass
    return jsonify({'ready': replicate_client.is_ready()})

@app.route('/api/ai/status')
@login_required
def api_ai_status():
    # Restore from session if available
    if not gemini_client.is_ready():
        key = session.get('gemini_key', '')
        if key:
            try:
                gemini_client.configure(key)
            except Exception:
                pass
    return jsonify({'ready': gemini_client.is_ready()})


@app.route('/api/ai/explain/prediction', methods=['POST'])
@login_required
def api_ai_explain_prediction():
    if not gemini_client.is_ready():
        key = session.get('gemini_key', '')
        if key:
            try: gemini_client.configure(key)
            except Exception: pass
    if not gemini_client.is_ready():
        return jsonify({'error': 'Gemini chưa cấu hình. Vui lòng nhập API key.'}), 503
    data = request.get_json(silent=True) or {}
    try:
        text = gemini_client.explain_prediction(
            drug_name    = data.get('drug_name', ''),
            disease_name = data.get('disease_name', ''),
            score        = float(data.get('score', 0)),
            is_known     = bool(data.get('is_known', False)),
            fuzzy_details= data.get('fuzzy_details'),
        )
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/explain/matrix', methods=['POST'])
@login_required
def api_ai_explain_matrix():
    if not gemini_client.is_ready():
        key = session.get('gemini_key', '')
        if key:
            try: gemini_client.configure(key)
            except Exception: pass
    if not gemini_client.is_ready():
        return jsonify({'error': 'Gemini chưa cấu hình.'}), 503
    data = request.get_json(silent=True) or {}
    try:
        text = gemini_client.explain_matrix(
            drugs    = data.get('drugs', []),
            diseases = data.get('diseases', []),
            matrix   = data.get('matrix', []),
            model_name = data.get('model', 'Fuzzy Logic'),
        )
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/explain/graph', methods=['POST'])
@login_required
def api_ai_explain_graph():
    if not gemini_client.is_ready():
        key = session.get('gemini_key', '')
        if key:
            try: gemini_client.configure(key)
            except Exception: pass
    if not gemini_client.is_ready():
        return jsonify({'error': 'Gemini chưa cấu hình.'}), 503
    data = request.get_json(silent=True) or {}
    try:
        text = gemini_client.explain_graph(
            entity_type = data.get('entity_type', 'drug'),
            entity_name = data.get('entity_name', ''),
            neighbors   = data.get('neighbors', []),
            model_name  = data.get('model', 'Fuzzy Logic'),
        )
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/explain/molecule', methods=['POST'])
@login_required
def api_ai_explain_molecule():
    if not gemini_client.is_ready():
        key = session.get('gemini_key', '')
        if key:
            try: gemini_client.configure(key)
            except Exception: pass
    if not gemini_client.is_ready():
        return jsonify({'error': 'Gemini chưa cấu hình.'}), 503
    data = request.get_json(silent=True) or {}
    try:
        text = gemini_client.explain_molecule(
            disease_name = data.get('disease_name', ''),
            candidates   = data.get('candidates', []),
        )
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/explain/fuzzy_layer', methods=['POST'])
@login_required
def api_ai_explain_fuzzy_layer():
    """Gemini explanation of GNN+Fuzzy firing strengths for a drug-disease pair."""
    if not gemini_client.is_ready():
        key = session.get('gemini_key', '')
        if key:
            try: gemini_client.configure(key)
            except Exception: pass
    if not gemini_client.is_ready():
        return jsonify({'error': 'Gemini chưa cấu hình.'}), 503
    data = request.get_json(silent=True) or {}
    try:
        text = gemini_client.explain_fuzzy_layer(
            drug_name    = data.get('drug_name', ''),
            disease_name = data.get('disease_name', ''),
            score        = float(data.get('score', 0)),
            n_rules      = int(data.get('n_rules', 32)),
            top_rules    = data.get('top_rules', []),
        )
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/fuzzy_animation', methods=['POST'])
@login_required
def api_ai_fuzzy_animation():
    """Gemini one-liner description of the animated fuzzy canvas."""
    if not gemini_client.is_ready():
        key = session.get('gemini_key', '')
        if key:
            try: gemini_client.configure(key)
            except Exception: pass
    if not gemini_client.is_ready():
        return jsonify({'error': 'Gemini chưa cấu hình.'}), 503
    data = request.get_json(silent=True) or {}
    try:
        text = gemini_client.explain_fuzzy_animation(
            n_rules      = int(data.get('n_rules', 32)),
            drug_name    = data.get('drug_name', ''),
            disease_name = data.get('disease_name', ''),
        )
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Replicate AI Routes ─────────────────────────────────
@app.route('/api/molecule/replicate', methods=['POST'])
@login_required
def api_molecule_replicate():
    """
    Generate drug candidates using Llama-3 on Replicate.
    Falls back gracefully if Replicate is not configured.
    """
    if not replicate_client.is_ready():
        tok = session.get('replicate_token', '')
        if tok:
            try: replicate_client.configure(tok)
            except Exception: pass
    if not replicate_client.is_ready():
        return jsonify({'error': 'Replicate chưa cấu hình. Vui lòng thêm API token.'}), 503

    data        = request.get_json(silent=True) or {}
    disease_idx = data.get('disease_idx')
    n           = min(int(data.get('n', 6)), 10)
    if disease_idx is None:
        return jsonify({'error': 'Thiếu disease_idx'}), 400

    dis_info = engine.get_disease_info(int(disease_idx))
    if not dis_info:
        return jsonify({'error': 'Không tìm thấy bệnh'}), 404

    # Fetch known drug names for context
    known_drug_names = [
        engine.drug_names[i] for i in dis_info.get('known_drugs', [])[:5]
    ]

    try:
        candidates = replicate_client.generate_molecules_llm(
            disease_name  = dis_info['name'],
            known_drugs   = known_drug_names,
            n_candidates  = n,
        )
        # Attach fuzzy score using engine
        for c in candidates:
            if 'score' not in c:
                c['score'] = round(float(engine._fuzzy_disease_drug(int(disease_idx)).mean()), 4)
        return jsonify({'disease': dis_info, 'candidates': candidates, 'source': 'replicate'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/molecule/image', methods=['POST'])
@login_required
def api_molecule_image():
    """Generate SDXL concept art of a drug-disease molecule via Replicate."""
    if not replicate_client.is_ready():
        tok = session.get('replicate_token', '')
        if tok:
            try: replicate_client.configure(tok)
            except Exception: pass
    if not replicate_client.is_ready():
        return jsonify({'error': 'Replicate chưa cấu hình.'}), 503

    data         = request.get_json(silent=True) or {}
    disease_name = data.get('disease_name', '')
    mol_name     = data.get('molecule_name', 'Unknown')
    smiles       = data.get('smiles', '')
    try:
        url = replicate_client.generate_molecule_image(disease_name, mol_name, smiles)
        if not url:
            return jsonify({'error': 'Không tạo được ảnh'}), 500
        return jsonify({'image_url': url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fuzzy/firing_strengths', methods=['POST'])
@login_required
def api_fuzzy_firing_strengths():
    """
    Return per-rule fuzzy firing strengths from the GNN+Fuzzy model
    for a specific (drug, disease) pair.  Used by the fuzzy animation canvas.
    """
    data        = request.get_json(silent=True) or {}
    drug_idx    = data.get('drug_idx')
    disease_idx = data.get('disease_idx')
    if drug_idx is None or disease_idx is None:
        return jsonify({'error': 'Thiếu drug_idx hoặc disease_idx'}), 400
    strengths = engine.get_fuzzy_firing_strengths(int(drug_idx), int(disease_idx))
    if not strengths:
        return jsonify({'error': 'Mô hình GNN+Fuzzy chưa sẵn sàng'}), 503
    return jsonify({'firing_strengths': strengths, 'n_rules': len(strengths)})


@app.route('/api/replicate/fuzzy_visual', methods=['POST'])
@login_required
def api_replicate_fuzzy_visual():
    """Generate an SDXL artistic image of the fuzzy neural network processing."""
    if not replicate_client.is_ready():
        tok = session.get('replicate_token', '')
        if tok:
            try: replicate_client.configure(tok)
            except Exception: pass
    if not replicate_client.is_ready():
        return jsonify({'error': 'Replicate chưa cấu hình.'}), 503

    data         = request.get_json(silent=True) or {}
    drug_name    = data.get('drug_name', '')
    disease_name = data.get('disease_name', '')
    n_rules      = int(data.get('n_rules', 32))
    try:
        url = replicate_client.generate_fuzzy_visual_image(drug_name, disease_name, n_rules)
        if not url:
            return jsonify({'error': 'Không tạo được ảnh'}), 500
        return jsonify({'image_url': url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Init & Run ───────────────────────────────────────────────
def initialize():
    db.init_db()
    if not db.data_imported():
        print('Đang nhập dữ liệu từ CSV vào cơ sở dữ liệu...')
        db.import_drugs(engine.as_drugs_list())
        db.import_diseases(engine.as_diseases_list())
        db.import_proteins(engine.as_proteins_list())
        db.import_associations(engine.drdi_assoc)
        print(f'  ✓ {db.count_drugs()} thuốc')
        print(f'  ✓ {db.count_diseases()} bệnh')
        print(f'  ✓ {db.count_proteins()} protein')
        print(f'  ✓ {db.count_associations()} liên kết đã biết')
    else:
        print('Dữ liệu đã được nhập trước đó.')


if __name__ == '__main__':
    initialize()
    app.run(debug=False, host='0.0.0.0', port=5000)
