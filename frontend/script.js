// ── Constants ──
const CIRCUMFERENCE = 314;
let running = false, paused = false, progress = 0, stepIdx = -1;
let interval = null, logInterval = null;

const stepMsgs = [
  ["iniciando extração...", "lendo kaggle_salaries.csv (4.2 MB)", "conectando remotive api...", "registros brutos: 12.847", "extraindo wef_future_of_jobs.pdf (198 pág.)"],
  ["iniciando limpeza...", "removendo 312 registros nulos", "corrigindo tipos: salary → float", "drop_duplicates(): -89 linhas", "dataset limpo: 12.446 registros"],
  ["padronizando moedas → USD", "normalizando 63 títulos de cargo", "engenharia de features: seniority_num", "skills_list gerado com 1.204 termos", "dataset_final.csv exportado ✓"],
  ["separando train/test (80/20)", "treinando RandomForestRegressor...", "n_estimators=100 | max_depth=None", "MAE: $8.240 | RMSE: $14.780 | R²: 0.83", "modelo salvo → models/salary_model.pkl ✓"]
];

const stepDurations = [6000, 5000, 6000, 7000];

// ── DOM Helpers ──
function getEl(id) { return document.getElementById(id); }

// ── Ring Progress ──
function setRing(pct) {
  const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
  getEl('ringProgress').style.strokeDashoffset = offset;
}

// ── Pulse Ring ──
function setPulse(on) {
  const p = getEl('ringPulse');
  if (on) {
    p.style.animation = 'spinSlow 3s linear infinite';
    p.style.opacity = '0.4';
  } else {
    p.style.animation = 'none';
    p.style.opacity = '0';
  }
}

// ── Step State ──
function setStep(idx, state) {
  const el = getEl('s' + idx);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (state) el.classList.add(state);
}

// ── Log ──
function appendLog(msg, cls = '') {
  const container = getEl('logLines');
  const div = document.createElement('div');
  div.className = 'log-line' + (cls ? ' ' + cls : '');
  div.textContent = msg;
  container.appendChild(div);
  while (container.children.length > 5) container.removeChild(container.firstChild);
  container.scrollTop = container.scrollHeight;
}

// ── Icons (Unicode) ──
const ICONS = {
  play: '\u25B6',
  pause: '\u23F8',
  check: '\u2713'
};

function setIcon(type) {
  getEl('btnIcon').textContent = ICONS[type] || ICONS.play;
}

// ── Pipeline Control ──
function startPipeline() {
  running = true;
  paused = false;
  stepIdx = 0;
  progress = 0;

  setIcon('pause');
  getEl('btnLabel').textContent = 'pausar';
  getEl('ringProgress').style.stroke = '#1D9E75';
  setPulse(false);
  setStep(0, 'active');
  getEl('statusTxt').innerHTML = 'status: <span class="st-running">executando</span>';
  appendLog('pipeline iniciado...', 'hi');
  runStep();
}

function runStep() {
  if (stepIdx >= 4) { finishPipeline(); return; }

  const dur = stepDurations[stepIdx] || 6000;
  const totalDur = stepDurations.reduce((a, b) => a + b, 0);
  const startPct = stepDurations.slice(0, stepIdx).reduce((a, b) => a + b, 0) / totalDur * 100;
  const endPct = stepDurations.slice(0, stepIdx + 1).reduce((a, b) => a + b, 0) / totalDur * 100;
  const msgs = stepMsgs[stepIdx] || [];
  let elapsed = 0;
  const tick = 80;

  clearInterval(interval);
  clearInterval(logInterval);

  let msgIdx = 0;
  logInterval = setInterval(() => {
    if (!running || paused) return;
    if (msgIdx < msgs.length) {
      appendLog(msgs[msgIdx], 'md');
      msgIdx++;
    }
  }, dur / (msgs.length + 1));

  interval = setInterval(() => {
    if (!running || paused) return;
    elapsed += tick;
    const localPct = elapsed / dur;
    progress = startPct + localPct * (endPct - startPct);
    setRing(progress);
    getEl('pctTxt').textContent = Math.round(progress) + '%';

    if (elapsed >= dur) {
      clearInterval(interval);
      clearInterval(logInterval);
      setStep(stepIdx, 'done');
      stepIdx++;
      if (stepIdx < 4) setStep(stepIdx, 'active');
      runStep();
    }
  }, tick);
}

function pausePipeline() {
  paused = true;
  running = false;
  setIcon('play');
  getEl('btnLabel').textContent = 'retomar';
  getEl('ringProgress').style.stroke = '#085041';
  getEl('statusTxt').innerHTML = 'status: <span class="st-paused">pausado</span>';
  appendLog('pausado pelo usuário.', 'md');
  setPulse(false);
}

function resumePipeline() {
  paused = false;
  running = true;
  setIcon('pause');
  getEl('btnLabel').textContent = 'pausar';
  getEl('ringProgress').style.stroke = '#1D9E75';
  getEl('statusTxt').innerHTML = 'status: <span class="st-running">executando</span>';
  appendLog('retomando...', 'hi');
  runStep();
}

function finishPipeline() {
  running = false;
  paused = false;
  progress = 100;
  setRing(100);
  getEl('pctTxt').textContent = '100%';
  setIcon('check');
  getEl('btnLabel').textContent = 'concluído';
  getEl('ringProgress').style.stroke = '#1D9E75';
  getEl('statusTxt').innerHTML = 'status: <span class="st-done">concluído ✓</span>';
  appendLog('pipeline finalizado com sucesso ✓', 'hi');
  for (let i = 0; i < 4; i++) setStep(i, 'done');
}

function resetPipeline() {
  progress = 0;
  stepIdx = -1;
  setRing(0);
  getEl('pctTxt').textContent = '0%';
  for (let i = 0; i < 4; i++) setStep(i, '');
  setIcon('play');
  getEl('btnLabel').textContent = 'iniciar';
  getEl('ringProgress').style.stroke = '#1D9E75';
  getEl('statusTxt').innerHTML = 'status: <span>aguardando</span>';
  getEl('logLines').innerHTML = '<div class="log-line">pipeline resetado. aguardando inicialização<span class="cursor">_</span></div>';
  setPulse(true);
}

function togglePipeline() {
  if (!running && !paused && stepIdx === -1) { startPipeline(); return; }
  if (running && !paused) { pausePipeline(); return; }
  if (paused) { resumePipeline(); return; }
  if (!running && progress === 100) { resetPipeline(); return; }
}

// ── Neural Network Background ──
function initNet() {
  const canvas = getEl('netCanvas');
  const ctx = canvas.getContext('2d');
  let W, H, nodes = [];

  function resize() {
    const r = canvas.parentElement.getBoundingClientRect();
    W = canvas.width = r.width;
    H = canvas.height = r.height || 560;
  }

  resize();

  for (let i = 0; i < 38; i++) {
    nodes.push({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 1
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    nodes.forEach(n => {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 130) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(29,158,117,${(1 - d / 130) * 0.5})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(29,158,117,.6)';
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  draw();
  window.addEventListener('resize', resize);
}

// ── Init ──
initNet();
setPulse(true);

getEl('mainBtn').addEventListener('click', togglePipeline);
getEl('mainBtn').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    togglePipeline();
  }
});
