import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine,
  ResponsiveContainer, Cell,
} from 'recharts';
import { downloadReport } from '../services/api';

const STAGE_CONFIG = {
  No_COVID:  { cls: 'no-covid',  icon: '✓',  label: 'No COVID-19 Detected' },
  Mild:      { cls: 'mild',      icon: '⚠',  label: 'Mild COVID-19' },
  Moderate:  { cls: 'moderate',  icon: '⚡',  label: 'Moderate COVID-19' },
  Severe:    { cls: 'severe',    icon: '🚨',  label: 'Severe COVID-19' },
  Critical:  { cls: 'critical',  icon: '🆘',  label: 'Critical — Emergency Care' },
};

const SHAP_COLORS = {
  positive: '#DC2626',  // pushes toward current severity
  negative: '#16A34A',  // pushes away from current severity
};

function SHAPChart({ features }) {
  if (!features || features.length === 0) return null;

  const data = [...features]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .map(f => ({
      feature: f.feature,
      value: parseFloat(f.shap_value.toFixed(4)),
    }));

  const maxAbs = Math.max(...data.map(d => Math.abs(d.value)), 0.01);

  return (
    <div className="shap-chart-wrapper">
      <ResponsiveContainer width="100%" height={data.length * 36 + 20}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 30, left: 0, bottom: 4 }}
        >
          <XAxis
            type="number"
            domain={[-maxAbs * 1.1, maxAbs * 1.1]}
            tickFormatter={v => v.toFixed(3)}
            tick={{ fontSize: 10, fill: '#64748B' }}
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={160}
            tick={{ fontSize: 11, fill: '#0F172A' }}
          />
          <Tooltip
            formatter={(v) => [v.toFixed(4), 'SHAP value']}
            contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #E2E8F0' }}
          />
          <ReferenceLine x={0} stroke="#E2E8F0" strokeWidth={1.5} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.value >= 0 ? SHAP_COLORS.positive : SHAP_COLORS.negative}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="shap-legend">
        <span><span className="shap-dot" style={{ background: SHAP_COLORS.positive }} />Increases severity</span>
        <span><span className="shap-dot" style={{ background: SHAP_COLORS.negative }} />Decreases severity</span>
      </div>
    </div>
  );
}

export default function DiagnosisResult({ result, patientInput, onReset }) {
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState(null);

  const config = STAGE_CONFIG[result.severity] || {
    cls: 'mild', icon: '?', label: result.severity,
  };

  const patientLabel = result.patient_name
    ? result.patient_id
      ? `${result.patient_name} (ID: ${result.patient_id})`
      : result.patient_name
    : result.patient_id
      ? `ID: ${result.patient_id}`
      : null;

  const handleDownloadPdf = async () => {
    setPdfLoading(true);
    setPdfError(null);
    try {
      const reportData = {
        patient_name:           result.patient_name,
        patient_id:             result.patient_id,
        timestamp:              result.timestamp,
        severity:               result.severity,
        confidence:             result.confidence,
        shap_features:          result.shap_features,
        llm_explanation:        result.llm_explanation,
        treatment_recommendation: result.treatment_recommendation,
        model_used:             result.model_used,
        // patient demographics from stored input
        age:             patientInput?.age    || 0,
        sex:             patientInput?.sex    || '',
        pregnant:        patientInput?.pregnant       || false,
        hospitalized:    patientInput?.hospitalized   || false,
        pneumonia:       patientInput?.pneumonia      || false,
        diabetes:        patientInput?.diabetes       || false,
        copd:            patientInput?.copd           || false,
        asthma:          patientInput?.asthma         || false,
        immunocompromised: patientInput?.immunocompromised || false,
        hypertension:    patientInput?.hypertension   || false,
        cardiovascular:  patientInput?.cardiovascular || false,
        obesity:         patientInput?.obesity        || false,
        renal_chronic:   patientInput?.renal_chronic  || false,
        tobacco:         patientInput?.tobacco        || false,
      };
      await downloadReport(reportData);
    } catch (err) {
      setPdfError(err.message);
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div className="result-card">

      {/* ── Severity header ── */}
      <div className={`result-header ${config.cls}`}>
        <span className={`severity-badge ${config.cls}`}>
          <span>{config.icon}</span>
          {config.label}
        </span>
        <span className="confidence-badge">
          Confidence: {(result.confidence * 100).toFixed(1)}%
        </span>
      </div>

      {/* ── Model mode badge ── */}
      <div style={{ padding: '8px 20px 0' }}>
        {result.model_mode === 'ensemble' ? (
          <span className="model-badge model-badge--ensemble">Ensemble Model (Clinical + Lab)</span>
        ) : (
          <span className="model-badge model-badge--comorbidity">Comorbidity Model</span>
        )}
      </div>

      {/* ── Patient + timestamp ── */}
      {(patientLabel || result.timestamp) && (
        <div className="result-patient-row">
          {patientLabel && (
            <span className="result-patient-name">Assessment for: {patientLabel}</span>
          )}
          {result.timestamp && (
            <span className="result-timestamp">{result.timestamp}</span>
          )}
        </div>
      )}

      <div className="result-body">

        {/* Treatment */}
        <div className="result-section">
          <div className="result-section-title">Treatment Recommendation</div>
          <div className={`prescription-box ${config.cls === 'critical' || config.cls === 'severe' ? 'critical' : ''}`}>
            {result.treatment_recommendation}
          </div>
        </div>

        {/* SHAP chart */}
        {result.shap_features && result.shap_features.length > 0 && (
          <div className="result-section">
            <div className="result-section-title">
              Top Contributing Factors
              <span className="shap-label-badge">SHAP explainability</span>
            </div>
            <SHAPChart features={result.shap_features} />
          </div>
        )}

        {/* AI Explanation */}
        {result.llm_explanation && (
          <div className="result-section">
            <div className="result-section-title">
              AI Clinical Explanation
              <span className="model-inline-badge">via {result.model_used}</span>
            </div>
            <p className="reasoning-text">{result.llm_explanation}</p>
          </div>
        )}

        {/* Download PDF */}
        <div className="result-section">
          {pdfError && (
            <div className="error-banner" style={{ marginBottom: 10 }}>
              <span>⚠</span> {pdfError}
            </div>
          )}
          <button
            className="btn-download"
            onClick={handleDownloadPdf}
            disabled={pdfLoading}
          >
            {pdfLoading ? (
              <>
                <span className="spinner spinner--dark" />
                Generating PDF...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7 10 12 15 17 10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Download PDF Report
              </>
            )}
          </button>
        </div>
      </div>

      {/* ── Footer ── */}
      <div className="result-footer">
        <span className="model-badge">
          <span className="model-dot" />
          GradientBoostingClassifier · {result.model_used}
        </span>
        <button
          className="btn-secondary"
          onClick={onReset}
          style={{ height: 36, padding: '0 18px', fontSize: '0.82rem' }}
        >
          New Assessment
        </button>
      </div>
    </div>
  );
}
