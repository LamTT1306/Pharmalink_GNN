/* main.js — shared utilities for AMDGT web app */

/**
 * Show a Bootstrap-style toast notification.
 * @param {string} message
 * @param {string} [type='success'] - Bootstrap color: success, danger, warning, info
 */
function showToast(message, type = 'success') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const id = 'toast-' + Date.now();
  const icons = { success: 'check-circle-fill', danger: 'x-circle-fill', warning: 'exclamation-triangle-fill', info: 'info-circle-fill' };
  const icon = icons[type] || icons.info;
  const html = `
    <div id="${id}" class="toast align-items-center text-bg-${type} border-0 show mb-2" role="alert" aria-live="assertive">
      <div class="d-flex">
        <div class="toast-body"><i class="bi bi-${icon} me-2"></i>${message}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    </div>`;
  container.insertAdjacentHTML('beforeend', html);
  const el = document.getElementById(id);
  const t = new bootstrap.Toast(el, { delay: 3000 });
  t.show();
  el.addEventListener('hidden.bs.toast', () => el.remove());
}
