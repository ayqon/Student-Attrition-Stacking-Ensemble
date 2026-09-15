// OULAD Analytics Platform - Interactive Logic
document.addEventListener('DOMContentLoaded', async () => {
  let appData = null;
  let pcaScatterChart = null;
  let dbscanChart = null;
  let stackingChart = null;

  // 1. Fetch Real Data
  try {
    const res = await fetch('real_data.json');
    appData = await res.json();
    console.log('Loaded OULAD Analytics Data:', appData);
  } catch (err) {
    console.error('Failed to load real_data.json:', err);
    return;
  }

  // 2. Tab Switching with layout reflow detection
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add('active');

      setTimeout(() => {
        if (btn.dataset.tab === 'tab-kmeans') {
          if (!pcaScatterChart) {
            initPcaScatter();
          } else {
            pcaScatterChart.resize();
            pcaScatterChart.update();
          }
        } else if (btn.dataset.tab === 'tab-dbscan' && dbscanChart) {
          dbscanChart.resize();
          dbscanChart.update();
        } else if (btn.dataset.tab === 'tab-stacking' && stackingChart) {
          stackingChart.resize();
          stackingChart.update();
        }
      }, 50);
    });
  });

  // 3. Top Metrics
  document.getElementById('m-students').textContent = appData.summary.total_students.toLocaleString();
  document.getElementById('m-acc').textContent = appData.summary.ensemble_accuracy.toFixed(2) + '%';
  document.getElementById('m-pca').textContent = appData.summary.pca_variance_retained.toFixed(1) + '%';
  document.getElementById('m-risk').textContent = appData.summary.at_risk_fail_rate.toFixed(1) + '%';
  document.getElementById('m-success').textContent = appData.summary.engaged_fail_rate.toFixed(1) + '%';

  // 4. TAB 1: Simulator
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

    let zPass = (coreComp * 3.5) + (attemptScore * 0.04) + (activeWeeks * 0.08) + (perfEff * 1.5) - (prevAttempts * 0.8) - 2.8;
    let zFail = -(coreComp * 2.8) - (attemptScore * 0.03) + (prevAttempts * 1.2) - (perfEff * 0.8) + 1.2;
    let zWithdrawn = -(activeWeeks * 0.12) - (activeDays * 0.03) + (studiedCredits * 0.005) + (10 - imdBand) * 0.1;

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
      predOutcomeEl.style.color = '#059669';
      predDescEl.textContent = 'High active engagement and assessment completion ratios indicate solid trajectory toward module completion.';
      advTitleEl.textContent = 'Standard Progress Monitoring';
      advTextEl.textContent = 'Maintain regular automated VLE check-ins and standard submission reminders.';
    } else if (probFail >= probPass && probFail >= probWith) {
      predOutcomeEl.textContent = 'FAIL (Academic Risk)';
      predOutcomeEl.style.color = '#dc2626';
      predDescEl.textContent = 'Low core score attempts and suboptimal performance efficiency place this student in the high academic failure hazard group.';
      advTitleEl.textContent = 'Urgent Academic Tutoring Referral';
      advTextEl.textContent = 'Trigger early intervention: schedule 1-on-1 tutoring review before upcoming assessment deadlines.';
    } else {
      predOutcomeEl.textContent = 'WITHDRAWN (Early Drop-off Risk)';
      predOutcomeEl.style.color = '#475569';
      predDescEl.textContent = 'Severe engagement collapse (low active weeks and active days) signals impending module attrition.';
      advTitleEl.textContent = 'Pastoral Re-engagement Outreach';
      advTextEl.textContent = 'Initiate immediate student support outreach to investigate barriers to study and offer study load adjustments.';
    }
  }

  [sActiveWeeks, sCoreComp, sAttemptScore, sPerfEff, sActiveDays, sPrevAttempts, sStudiedCredits, sImdBand].forEach(el => {
    el.addEventListener('input', updateSimulator);
  });

  document.querySelectorAll('.quick-presets-row button').forEach(btn => {
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

  document.getElementById('btn-reset-simulator').addEventListener('click', () => {
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

  // 5. TAB 2: Robust Chart.js 2D PCA Scatter Chart
  function initPcaScatter() {
    const ctxScatter = document.getElementById('pca-scatter-chart').getContext('2d');
    const pts = appData.scatter_points;

    const group0Pts = pts.filter(p => p.cluster === 0).map(p => ({ x: p.pc1, y: p.pc2, result: p.result }));
    const group1Pts = pts.filter(p => p.cluster === 1).map(p => ({ x: p.pc1, y: p.pc2, result: p.result }));

    const centroid0 = { x: appData.centroids[0].pc1, y: appData.centroids[0].pc2 };
    const centroid1 = { x: appData.centroids[1].pc1, y: appData.centroids[1].pc2 };

    pcaScatterChart = new Chart(ctxScatter, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Group 0: At-Risk (40.4% Fail)',
            data: group0Pts,
            backgroundColor: 'rgba(220, 38, 38, 0.45)',
            borderColor: 'rgba(220, 38, 38, 0.7)',
            pointRadius: 3,
            pointHoverRadius: 6
          },
          {
            label: 'Group 1: Engaged (7.1% Fail)',
            data: group1Pts,
            backgroundColor: 'rgba(5, 150, 105, 0.45)',
            borderColor: 'rgba(5, 150, 105, 0.7)',
            pointRadius: 3,
            pointHoverRadius: 6
          },
          {
            label: 'Centroid 0 (At-Risk)',
            data: [centroid0],
            backgroundColor: '#dc2626',
            borderColor: '#ffffff',
            borderWidth: 2,
            pointRadius: 9,
            pointStyle: 'rectRot'
          },
          {
            label: 'Centroid 1 (Engaged)',
            data: [centroid1],
            backgroundColor: '#059669',
            borderColor: '#ffffff',
            borderWidth: 2,
            pointRadius: 9,
            pointStyle: 'rectRot'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11, family: 'Inter' } } },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: (PC1: ${ctx.raw.x.toFixed(2)}, PC2: ${ctx.raw.y.toFixed(2)})`
            }
          }
        },
        scales: {
          x: { title: { display: true, text: 'Principal Component 1 (Engagement Dimension)' }, grid: { color: '#f1f5f9' } },
          y: { title: { display: true, text: 'Principal Component 2 (Assessment Dimension)' }, grid: { color: '#f1f5f9' } }
        }
      }
    });

    // Toggle outcomes vs clusters
    document.getElementById('btn-toggle-clusters').addEventListener('click', (e) => {
      e.target.classList.add('active');
      document.getElementById('btn-toggle-outcomes').classList.remove('active');

      pcaScatterChart.data.datasets[0].label = 'Group 0: At-Risk (40.4% Fail)';
      pcaScatterChart.data.datasets[0].data = group0Pts;
      pcaScatterChart.data.datasets[0].backgroundColor = 'rgba(220, 38, 38, 0.45)';

      pcaScatterChart.data.datasets[1].label = 'Group 1: Engaged (7.1% Fail)';
      pcaScatterChart.data.datasets[1].data = group1Pts;
      pcaScatterChart.data.datasets[1].backgroundColor = 'rgba(5, 150, 105, 0.45)';

      pcaScatterChart.data.datasets[2].hidden = false;
      pcaScatterChart.data.datasets[3].hidden = false;
      pcaScatterChart.update();
    });

    document.getElementById('btn-toggle-outcomes').addEventListener('click', (e) => {
      e.target.classList.add('active');
      document.getElementById('btn-toggle-clusters').classList.remove('active');

      const passPts = pts.filter(p => p.result === 'Pass').map(p => ({ x: p.pc1, y: p.pc2 }));
      const failPts = pts.filter(p => p.result === 'Fail').map(p => ({ x: p.pc1, y: p.pc2 }));
      const withPts = pts.filter(p => p.result === 'Withdrawn').map(p => ({ x: p.pc1, y: p.pc2 }));

      pcaScatterChart.data.datasets[0].label = 'True Pass (59.6%)';
      pcaScatterChart.data.datasets[0].data = passPts;
      pcaScatterChart.data.datasets[0].backgroundColor = 'rgba(5, 150, 105, 0.5)';

      pcaScatterChart.data.datasets[1].label = 'True Fail (22.3%)';
      pcaScatterChart.data.datasets[1].data = failPts;
      pcaScatterChart.data.datasets[1].backgroundColor = 'rgba(220, 38, 38, 0.5)';

      pcaScatterChart.data.datasets[2].hidden = true;
      pcaScatterChart.data.datasets[3].hidden = true;
      pcaScatterChart.update();
    });
  }

  initPcaScatter();

  // 6. TAB 3: DBSCAN Bar Chart
  const ctxDbscan = document.getElementById('dbscan-bar-chart').getContext('2d');
  dbscanChart = new Chart(ctxDbscan, {
    type: 'bar',
    data: {
      labels: ['Core Dense Cohort (88.1%)', 'Noise Outliers Isolated (11.9%)'],
      datasets: [{
        label: 'Failure Rate (%)',
        data: [17.8, 68.4],
        backgroundColor: ['#059669', '#dc2626'],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 100, title: { display: true, text: 'Academic Failure Rate (%)' } },
        x: { grid: { display: false } }
      }
    }
  });

  // 7. TAB 4: Stacking Benchmark Chart
  const ctxStack = document.getElementById('stacking-chart').getContext('2d');
  stackingChart = new Chart(ctxStack, {
    type: 'bar',
    data: {
      labels: ['Stacking Ensemble', 'XGBoost', 'Random Forest', 'SVM (RBF)', 'KNN (Dropped)'],
      datasets: [{
        label: 'Classification Accuracy (%)',
        data: [81.09, 79.23, 78.45, 77.80, 71.20],
        backgroundColor: ['#2563eb', '#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0'],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 65, max: 85, title: { display: true, text: 'Accuracy (%)' } },
        x: { grid: { display: false } }
      }
    }
  });

});
