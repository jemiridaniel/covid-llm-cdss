const API_BASE = process.env.REACT_APP_API_URL || '';

export async function diagnosePatient(patientData) {
  const response = await fetch(`${API_BASE}/api/v1/diagnose`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patientData),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${response.status})`);
  }

  return response.json();
}

export async function downloadReport(reportData) {
  const response = await fetch(`${API_BASE}/api/v1/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reportData),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Report generation failed (${response.status})`);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const slug = reportData.patient_id || reportData.patient_name || 'patient';
  a.download = `covid-report-${slug.replace(/\s+/g, '-')}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
