import React, { useState } from 'react';

const COMORBIDITIES = [
  { key: 'diabetes',        label: 'Diabetes' },
  { key: 'copd',            label: 'COPD' },
  { key: 'asthma',          label: 'Asthma' },
  { key: 'immunocompromised', label: 'Immunocompromised' },
  { key: 'hypertension',    label: 'Hypertension' },
  { key: 'cardiovascular',  label: 'Cardiovascular disease' },
  { key: 'obesity',         label: 'Obesity' },
  { key: 'renal_chronic',   label: 'Chronic kidney disease' },
  { key: 'tobacco',         label: 'Tobacco use' },
];

const DEFAULT_FORM = {
  age: '',
  sex: 'male',
  pregnant: false,
  pneumonia: false,
  hospitalized: false,
  diabetes: false,
  copd: false,
  asthma: false,
  immunocompromised: false,
  hypertension: false,
  cardiovascular: false,
  obesity: false,
  renal_chronic: false,
  tobacco: false,
  patient_name: '',
  patient_id: '',
  // Lab results (optional — Einstein ensemble)
  lab_hemoglobin: '',
  lab_leukocytes: '',
  lab_lymphocytes: '',
  lab_platelets: '',
  lab_crp: '',
  lab_ddimer: '',
  lab_ferritin: '',
  lab_ldh: '',
};

const MILD_PRESET = {
  ...DEFAULT_FORM,
  age: 28,
  sex: 'male',
  pregnant: false,
  pneumonia: false,
  hospitalized: false,
};

const CRITICAL_PRESET = {
  ...DEFAULT_FORM,
  age: 75,
  sex: 'male',
  pregnant: false,
  pneumonia: true,
  hospitalized: true,
  diabetes: true,
  hypertension: true,
  obesity: true,
  cardiovascular: true,
};

export default function DiagnosisForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [showLab, setShowLab] = useState(false);

  const set = (key, value) => setForm(prev => ({ ...prev, [key]: value }));
  const toggle = (key) => setForm(prev => ({ ...prev, [key]: !prev[key] }));

  const activeComorbidities = COMORBIDITIES.filter(c => form[c.key]).length;

  const handleSubmit = (e) => {
    e.preventDefault();

    const labEntries = showLab ? {
      hemoglobin:  form.lab_hemoglobin,
      leukocytes:  form.lab_leukocytes,
      lymphocytes: form.lab_lymphocytes,
      platelets:   form.lab_platelets,
      crp:         form.lab_crp,
      ddimer:      form.lab_ddimer,
      ferritin:    form.lab_ferritin,
      ldh:         form.lab_ldh,
    } : {};

    const hasLab = Object.values(labEntries).some(v => v !== '');

    onSubmit({
      ...form,
      age: parseInt(form.age, 10) || 0,
      lab_values: hasLab ? labEntries : null,
    });
  };

  const applyPreset = (preset) => setForm(preset);
  const handleReset = () => {
    setForm(DEFAULT_FORM);
    setShowLab(false);
  };

  return (
    <form onSubmit={handleSubmit}>

      {/* ── Demo Presets ── */}
      <div className="demo-presets">
        <span className="demo-presets-label">Try a demo:</span>
        <button type="button" className="btn-preset" onClick={() => applyPreset(MILD_PRESET)}>
          Try Mild Case
        </button>
        <button type="button" className="btn-preset btn-preset--critical" onClick={() => applyPreset(CRITICAL_PRESET)}>
          Try Critical Case
        </button>
      </div>

      {/* ── Patient Identification ── */}
      <div className="card">
        <div className="card-header">
          <h2>Patient Identification <span className="optional-tag">optional</span></h2>
        </div>
        <div className="card-body">
          <div className="patient-id-grid">
            <div className="field field--large">
              <label>Patient Name</label>
              <input
                type="text"
                placeholder="e.g. John Doe"
                value={form.patient_name}
                onChange={e => set('patient_name', e.target.value)}
              />
            </div>
            <div className="field field--large">
              <label>Patient ID</label>
              <input
                type="text"
                placeholder="e.g. CVD-001"
                value={form.patient_id}
                onChange={e => set('patient_id', e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── Demographics ── */}
      <div className="card">
        <div className="card-header">
          <h2>Patient Demographics</h2>
        </div>
        <div className="card-body">
          <div className="demographics-grid">
            <div className="field">
              <label>Age</label>
              <input
                type="number"
                min="0"
                max="120"
                placeholder="e.g. 45"
                value={form.age}
                onChange={e => set('age', e.target.value)}
                required
              />
            </div>
            <div className="field">
              <label>Sex</label>
              <select value={form.sex} onChange={e => set('sex', e.target.value)}>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
            {form.sex === 'female' && (
              <div className="field">
                <label>Pregnant</label>
                <label className={`toggle-checkbox ${form.pregnant ? 'checked' : ''}`}>
                  <input
                    type="checkbox"
                    checked={form.pregnant}
                    onChange={() => toggle('pregnant')}
                  />
                  <span>{form.pregnant ? 'Yes' : 'No'}</span>
                </label>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Clinical Status ── */}
      <div className="card">
        <div className="card-header">
          <h2>Clinical Status</h2>
        </div>
        <div className="card-body">
          <div className="symptoms-grid">
            <label className={`symptom-checkbox ${form.pneumonia ? 'checked' : ''}`}>
              <input
                type="checkbox"
                checked={form.pneumonia}
                onChange={() => toggle('pneumonia')}
              />
              <span>Pneumonia confirmed</span>
            </label>
            <label className={`symptom-checkbox ${form.hospitalized ? 'checked' : ''}`}>
              <input
                type="checkbox"
                checked={form.hospitalized}
                onChange={() => toggle('hospitalized')}
              />
              <span>Currently hospitalized</span>
            </label>
          </div>
        </div>
      </div>

      {/* ── Comorbidities ── */}
      <div className="card">
        <div className="card-header">
          <h2>
            Comorbidities
            {activeComorbidities > 0 && (
              <span style={{ color: '#EA580C', fontWeight: 400, fontSize: '0.82rem', marginLeft: 8 }}>
                ({activeComorbidities} selected)
              </span>
            )}
          </h2>
        </div>
        <div className="card-body">
          <div className="symptoms-grid">
            {COMORBIDITIES.map(({ key, label }) => (
              <label
                key={key}
                className={`risk-factor-checkbox ${form[key] ? 'checked' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={form[key]}
                  onChange={() => toggle(key)}
                />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* ── Lab Results (optional) ── */}
      <div className="card">
        <div className="card-header" style={{ cursor: 'pointer' }} onClick={() => setShowLab(v => !v)}>
          <h2>
            Lab Results <span className="optional-tag">optional</span>
            <span style={{ marginLeft: 8, fontSize: '0.8rem', color: '#6B7280' }}>
              {showLab ? '▲ hide' : '▼ Adding lab results activates ensemble model'}
            </span>
          </h2>
        </div>
        {showLab && (
          <div className="card-body">
            <p className="lab-info-text">Activates ensemble model for improved accuracy (Einstein Hospital lab features)</p>
            <div className="demographics-grid">
              <div className="field">
                <label>Hemoglobin (g/dL)</label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 13.5"
                  value={form.lab_hemoglobin}
                  onChange={e => set('lab_hemoglobin', e.target.value)}
                />
              </div>
              <div className="field">
                <label>Leukocytes</label>
                <input
                  type="number"
                  step="0.01"
                  placeholder="e.g. 7.2"
                  value={form.lab_leukocytes}
                  onChange={e => set('lab_leukocytes', e.target.value)}
                />
              </div>
              <div className="field">
                <label>Lymphocytes (%)</label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 25.0"
                  value={form.lab_lymphocytes}
                  onChange={e => set('lab_lymphocytes', e.target.value)}
                />
              </div>
              <div className="field">
                <label>Platelets</label>
                <input
                  type="number"
                  placeholder="e.g. 180"
                  value={form.lab_platelets}
                  onChange={e => set('lab_platelets', e.target.value)}
                />
              </div>
              <div className="field">
                <label>CRP (mg/dL)</label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 5.2"
                  value={form.lab_crp}
                  onChange={e => set('lab_crp', e.target.value)}
                />
              </div>
              <div className="field">
                <label>D-Dimer</label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 0.5"
                  value={form.lab_ddimer}
                  onChange={e => set('lab_ddimer', e.target.value)}
                />
              </div>
              <div className="field">
                <label>Ferritin</label>
                <input
                  type="number"
                  placeholder="e.g. 200"
                  value={form.lab_ferritin}
                  onChange={e => set('lab_ferritin', e.target.value)}
                />
              </div>
              <div className="field">
                <label>LDH</label>
                <input
                  type="number"
                  placeholder="e.g. 250"
                  value={form.lab_ldh}
                  onChange={e => set('lab_ldh', e.target.value)}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Actions ── */}
      <div className="submit-row">
        <button type="submit" className="btn-primary" disabled={loading || !form.age}>
          {loading ? (
            <>
              <span className="spinner" />
              Analyzing...
            </>
          ) : (
            'Run ML Assessment'
          )}
        </button>
        <button type="button" className="btn-secondary" onClick={handleReset} disabled={loading}>
          Reset
        </button>
      </div>
    </form>
  );
}
