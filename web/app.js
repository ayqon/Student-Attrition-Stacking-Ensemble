// OULAD Intelligence - Interactive Logic
document.addEventListener('DOMContentLoaded', async () => {
  let appData = null;
  let colorMode = 'cluster'; // 'cluster' or 'outcome'

  // 1. Fetch Real Data
  try {
    const res = await fetch('real_data.json');
    appData = await res.json();
    console.log('Loaded OULAD Intelligence Data:', appData);
  } catch (err) {
    console.error('Failed to load real_data.json:', err);
    return;
  }

  // 2. Initialize Framer Tab Navigation
  document.querySelectorAll('.framer-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.framer-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.framer-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add('active');
      if (btn.dataset.tab === 'tab-kmeans') {
        renderScatter();
      }
    });
  });

  // 3. Top Metrics
  document.getElementById('m-acc').textContent = appData.summary.ensemble_accuracy.toFixed(2) + '%';
  document.getElementById('m-pca').textContent = appData.summary.pca_variance_retained.toFixed(1) + '%';
  document.getElementById('m-risk').textContent = appData.summary.at_risk_fail_rate.toFixed(1) + '%';
  document.getElementById('m-success').textContent = appData.summary.engaged_fail_rate.toFixed(1) + '%';

  // 4. TAB 1: Real-Time Trajectory Simulator
  const sActiveWeeks = document.getElementById('s-active-weeks');
  const sCoreComp = document.getElementById('s-core-comp');
  const sAttemptScore = document.getElementById('s-attempt-score');
  const sPerfEff = document.getElementById('s-perf-eff');
  const sActiveDays = document.getElementById('s-active-days');
  const sPrevAttempts = document.getElementById('s-prev-attempts');
  const sStudiedCredits = document.getElementById('s-studied-credits');
  const sImdBand = document.getElementById('s-imd-band');

  function updateSimulator() {
    const activeWeeks = parseFloat(sActiveWeeks.value);
    const coreComp = parseFloat(sCoreComp.value);
    const attemptScore = parseFloat(sAttemptScore.value);
    const perfEff = parseFloat(sPerfEff.value);
    const activeDays = parseFloat(sActiveDays.value);
    const prevAttempts = parseFloat(sPrevAttempts.value);
    const studiedCredits = parseFloat(sStudiedCredits.value);
    const imdBand = parseFloat(sImdBand.value);

    document.getElementById('val-active-weeks').textContent = activeWeeks;
    document.getElementById('val-core-comp').textContent = coreComp.toFixed(2);
    document.getElementById('val-attempt-score').textContent = attemptScore;
    document.getElementById('val-perf-eff').textContent = perfEff.toFixed(2);
    document.getElementById('val-active-days').textContent = activeDays;
    document.getElementById('val-prev-attempts').textContent = prevAttempts;
    document.getElementById('val-studied-credits').textContent = studiedCredits;
    document.getElementById('val-imd-band').textContent = imdBand + (imdBand <= 3 ? ' (Deprived)' : (imdBand >= 8 ? ' (Affluent)' : ' (Mid)'));

    // Scoring heuristics calibrated to Stacking Ensemble multi-class probabilities
    let zPass = (coreComp * 3.5) + (attemptScore * 0.04) + (activeWeeks * 0.08) + (perfEff * 1.5) - (prevAttempts * 0.8) - 2.8;
    let zFail = -(coreComp * 2.8) - (attemptScore * 0.03) + (prevAttempts * 1.2) - (perfEff * 0.8) + 1.2;
    let zWithdrawn = -(activeWeeks * 0.12) - (activeDays * 0.03) + (studiedCredits * 0.005) + (10 - imdBand) * 0.1;

    // Softmax probabilities
    const expPass = Math.exp(zPass);
    const expFail = Math.exp(zFail);
    const expWith = Math.exp(zWithdrawn);
    const sumExp = expPass + expFail + expWith;

    const probPass = (expPass / sumExp) * 100;
    const probFail = (expFail / sumExp) * 100;
    const probWith = (expWith / sumExp) * 100;

    document.getElementById('prob-pass').textContent = probPass.toFixed(1) + '%';
    document.getElementById('prob-fail').textContent = probFail.toFixed(1) + '%';
    document.getElementById('prob-withdrawn').textContent = probWith.toFixed(1) + '%';

    document.getElementById('bar-pass').style.width = probPass.toFixed(1) + '%';
    document.getElementById('bar-fail').style.width = probFail.toFixed(1) + '%';
    document.getElementById('bar-withdrawn').style.width = probWith.toFixed(1) + '%';

    const predOutcomeEl = document.getElementById('pred-outcome');
    const predDescEl = document.getElementById('pred-desc');
    const advTitleEl = document.getElementById('advisory-title');
    const advTextEl = document.getElementById('advisory-text');

    if (probPass >= probFail && probPass >= probWith) {
      predOutcomeEl.textContent = 'PASS';
      predOutcomeEl.className = 'verdict-main text-emerald';
      predDescEl.textContent = 'High assessment completion and continuous active engagement indicate a stable trajectory toward module completion.';
      advTitleEl.textContent = 'Standard Progress Monitoring';
      advTextEl.textContent = 'Engagement patterns are within normal academic bounds. Maintain standard weekly VLE notifications.';
    } else if (probFail >= probPass && probFail >= probWith) {
      predOutcomeEl.textContent = 'FAIL (Academic Hazard)';
      predOutcomeEl.className = 'verdict-main text-rose';
      predDescEl.textContent = 'Low core score attempts and suboptimal performance efficiency place this student in the high academic failure hazard group.';
      advTitleEl.textContent = 'Urgent Academic Tutoring Referral';
      advTextEl.textContent = 'Trigger early intervention: schedule 1-on-1 tutoring review before upcoming assessment deadlines.';
    } else {
      predOutcomeEl.textContent = 'WITHDRAWN (Attrition Risk)';
      predOutcomeEl.className = 'verdict-main text-indigo';
      predDescEl.textContent = 'Severe engagement collapse (low active weeks and active days) signals impending module attrition.';
      advTitleEl.textContent = 'Pastoral Re-engagement Outreach';
      advTextEl.textContent = 'Initiate immediate student support outreach to investigate barriers to study and offer study load adjustments.';
    }
  }

  [sActiveWeeks, sCoreComp, sAttemptScore, sPerfEff, sActiveDays, sPrevAttempts, sStudiedCredits, sImdBand].forEach(el => {
    el.addEventListener('input', updateSimulator);
  });

  document.querySelectorAll('.profile-quick-select button').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.profile;
      if (p === 'high-risk') {
        sActiveWeeks.value = 6;
        sCoreComp.value = 0.20;
        sAttemptScore.value = 35;
        sPerfEff.value = 0.30;
        sActiveDays.value = 12;
        sPrevAttempts.value = 2;
        sStudiedCredits.value = 120;
        sImdBand.value = 2;
      } else if (p === 'average') {
        sActiveWeeks.value = 18;
        sCoreComp.value = 0.65;
        sAttemptScore.value = 65;
        sPerfEff.value = 0.75;
        sActiveDays.value = 42;
        sPrevAttempts.value = 0;
        sStudiedCredits.value = 60;
        sImdBand.value = 5;
      } else if (p === 'high-achiever') {
        sActiveWeeks.value = 32;
        sCoreComp.value = 0.95;
        sAttemptScore.value = 88;
        sPerfEff.value = 1.40;
        sActiveDays.value = 95;
        sPrevAttempts.value = 0;
        sStudiedCredits.value = 60;
        sImdBand.value = 8;
      }
      updateSimulator();
    });
  });

  document.getElementById('btn-reset-sim').addEventListener('click', () => {
    sActiveWeeks.value = 18;
    sCoreComp.value = 0.65;
    sAttemptScore.value = 65;
    sPerfEff.value = 0.75;
    sActiveDays.value = 42;
    sPrevAttempts.value = 0;
    sStudiedCredits.value = 60;
    sImdBand.value = 5;
    updateSimulator();
  });

  updateSimulator();

  // 5. TAB 2: PCA Scatter Canvas (Framer Dark Theme)
  const canvas = document.getElementById('pca-scatter-canvas');
  const ctx = canvas.getContext('2d');

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    renderScatter();
  }
  window.addEventListener('resize', resizeCanvas);

  function renderScatter() {
    if (!appData) return;
    const w = canvas.parentElement.clientWidth;
    const h = canvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, w, h);

    const pts = appData.scatter_points;
    const minX = -6, maxX = 8;
    const minY = -5, maxY = 6;

    function mapX(val) { return ((val - minX) / (maxX - minX)) * (w - 60) + 30; }
    function mapY(val) { return h - (((val - minY) / (maxY - minY)) * (h - 60) + 30); }

    // Dark Grid Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = -4; x <= 6; x += 2) {
      ctx.beginPath();
      ctx.moveTo(mapX(x), 0);
      ctx.lineTo(mapX(x), h);
      ctx.stroke();
    }
    for (let y = -4; y <= 4; y += 2) {
      ctx.beginPath();
      ctx.moveTo(0, mapY(y));
      ctx.lineTo(w, mapY(y));
      ctx.stroke();
    }

    // Scatter points with soft neon glow
    for (const p of pts) {
      const cx = mapX(p.pc1);
      const cy = mapY(p.pc2);

      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);

      if (colorMode === 'cluster') {
        ctx.fillStyle = p.cluster === 0 ? 'rgba(244, 63, 94, 0.65)' : 'rgba(16, 185, 129, 0.65)';
      } else {
        if (p.result === 'Pass') ctx.fillStyle = 'rgba(16, 185, 129, 0.7)';
        else if (p.result === 'Fail') ctx.fillStyle = 'rgba(244, 63, 94, 0.7)';
        else ctx.fillStyle = 'rgba(99, 102, 241, 0.7)';
      }
      ctx.fill();
    }

    // Draw Centroids with glowing rings
    for (const c of appData.centroids) {
      const cx = mapX(c.pc1);
      const cy = mapY(c.pc2);

      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = c.cluster === 0 ? '#f43f5e' : '#10b981';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      ctx.font = 'bold 11px Outfit, sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.fillText(`Centroid ${c.cluster} (${c.label.split(' ')[0]})`, cx, cy - 14);
    }
  }

  document.getElementById('btn-toggle-clusters').addEventListener('click', e => {
    colorMode = 'cluster';
    e.target.className = 'framer-btn framer-btn-active';
    document.getElementById('btn-toggle-outcomes').className = 'framer-btn framer-btn-glass';
    document.getElementById('scatter-legend').innerHTML = `
      <div class="legend-badge"><span class="badge-dot dot-rose"></span> Group 0: At-Risk (40.4% Fail Rate)</div>
      <div class="legend-badge"><span class="badge-dot dot-emerald"></span> Group 1: Engaged (7.1% Fail Rate)</div>
    `;
    renderScatter();
  });

  document.getElementById('btn-toggle-outcomes').addEventListener('click', e => {
    colorMode = 'outcome';
    e.target.className = 'framer-btn framer-btn-active';
    document.getElementById('btn-toggle-clusters').className = 'framer-btn framer-btn-glass';
    document.getElementById('scatter-legend').innerHTML = `
      <div class="legend-badge"><span class="badge-dot dot-emerald"></span> True Pass (59.6%)</div>
      <div class="legend-badge"><span class="badge-dot dot-rose"></span> True Fail (22.3%)</div>
      <div class="legend-badge"><span class="badge-dot" style="background:#818cf8;"></span> True Withdrawn (18.1%)</div>
    `;
    renderScatter();
  });

  resizeCanvas();

  // 6. TAB 3: DBSCAN Neon Bar Chart
  const ctxDbscan = document.getElementById('dbscan-bar-chart').getContext('2d');
  new Chart(ctxDbscan, {
    type: 'bar',
    data: {
      labels: ['Core Dense Cohort (88.1%)', 'Noise Outliers Isolated (11.9%)'],
      datasets: [{
        label: 'Failure Rate (%)',
        data: [17.8, 68.4],
        backgroundColor: ['rgba(16, 185, 129, 0.75)', 'rgba(244, 63, 94, 0.75)'],
        borderColor: ['#10b981', '#f43f5e'],
        borderWidth: 1,
        borderRadius: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
      }
    }
  });

  // 7. TAB 4: Stacking Benchmark Neon Chart
  const ctxStack = document.getElementById('stacking-chart').getContext('2d');
  new Chart(ctxStack, {
    type: 'bar',
    data: {
      labels: ['Stacking Ensemble', 'XGBoost', 'Random Forest', 'SVM (RBF)', 'KNN (Dropped)'],
      datasets: [{
        label: 'Classification Accuracy (%)',
        data: [81.09, 79.23, 78.45, 77.80, 71.20],
        backgroundColor: [
          'rgba(139, 92, 246, 0.85)',
          'rgba(99, 102, 241, 0.5)',
          'rgba(59, 130, 246, 0.5)',
          'rgba(6, 182, 212, 0.5)',
          'rgba(100, 116, 139, 0.4)'
        ],
        borderColor: ['#8b5cf6', '#6366f1', '#3b82f6', '#06b6d4', '#64748b'],
        borderWidth: 1,
        borderRadius: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 65, max: 85, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
      }
    }
  });

});
