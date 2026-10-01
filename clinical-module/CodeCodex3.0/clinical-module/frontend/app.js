/**
 * MedScope AI - Clinical Data Module Frontend Script
 * Handles real-time API communication, health monitoring, preset population,
 * and result explainability rendering.
 */

const API_BASE_URL = "http://127.0.0.1:8000";

let currentModule = "heart"; // 'heart' or 'stroke'

// --- Initialize on Page Load ---
document.addEventListener("DOMContentLoaded", () => {
  checkServiceHealth();
  setupFormListeners();
});

// --- Health Check Monitor ---
async function checkServiceHealth() {
  const dot = document.getElementById("healthDot");
  const text = document.getElementById("healthText");
  
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error("Service unavailable");
    
    const data = await response.json();
    if (data.status === "ok" && data.models.heart === "loaded" && data.models.stroke === "loaded") {
      dot.className = "status-dot online";
      text.textContent = "AI SERVICE ONLINE";
    } else {
      dot.className = "status-dot offline";
      text.textContent = "AI SERVICE PARTIAL";
    }
  } catch (err) {
    dot.className = "status-dot offline";
    text.textContent = "AI SERVICE OFFLINE";
  }
}

// --- Module Tab Switcher ---
function switchTab(module) {
  currentModule = module;
  
  const tabHeart = document.getElementById("tabHeart");
  const tabStroke = document.getElementById("tabStroke");
  const heartForm = document.getElementById("heartForm");
  const strokeForm = document.getElementById("strokeForm");
  const formTitle = document.getElementById("formTitle");
  const formDesc = document.getElementById("formDesc");
  
  if (module === "heart") {
    tabHeart.classList.add("active");
    tabStroke.classList.remove("active");
    heartForm.classList.remove("hidden");
    strokeForm.classList.add("hidden");
    formTitle.textContent = "Heart Disease Clinical Assessment";
    formDesc.textContent = "Input patient metrics to perform AI risk screening and feature importance evaluation.";
  } else {
    tabStroke.classList.add("active");
    tabHeart.classList.remove("active");
    strokeForm.classList.remove("hidden");
    heartForm.classList.add("hidden");
    formTitle.textContent = "Stroke Risk Assessment";
    formDesc.textContent = "Input demographic and clinical variables for non-coding stroke screening.";
  }
}

// --- Preset Data Fillers for Demo ---
function fillPreset(type) {
  if (currentModule === "heart") {
    if (type === "high") {
      document.getElementById("h_age").value = 67;
      document.getElementById("h_sex").value = 1;
      document.getElementById("h_cp").value = 0; // Typical Angina
      document.getElementById("h_trestbps").value = 160;
      document.getElementById("h_chol").value = 286;
      document.getElementById("h_fbs").value = 1;
      document.getElementById("h_restecg").value = 0;
      document.getElementById("h_thalach").value = 108;
      document.getElementById("h_exang").value = 1;
      document.getElementById("h_oldpeak").value = 1.5;
      document.getElementById("h_slope").value = 1;
      document.getElementById("h_ca").value = 3;
      document.getElementById("h_thal").value = 2;
    } else {
      document.getElementById("h_age").value = 41;
      document.getElementById("h_sex").value = 0;
      document.getElementById("h_cp").value = 2; // Non-anginal
      document.getElementById("h_trestbps").value = 130;
      document.getElementById("h_chol").value = 204;
      document.getElementById("h_fbs").value = 0;
      document.getElementById("h_restecg").value = 0;
      document.getElementById("h_thalach").value = 172;
      document.getElementById("h_exang").value = 0;
      document.getElementById("h_oldpeak").value = 0.0;
      document.getElementById("h_slope").value = 2;
      document.getElementById("h_ca").value = 0;
      document.getElementById("h_thal").value = 2;
    }
  } else if (currentModule === "stroke") {
    if (type === "high") {
      document.getElementById("s_gender").value = "Male";
      document.getElementById("s_age").value = 79.0;
      document.getElementById("s_hypertension").value = 1;
      document.getElementById("s_heart_disease").value = 1;
      document.getElementById("s_ever_married").value = "Yes";
      document.getElementById("s_work_type").value = "Self-employed";
      document.getElementById("s_Residence_type").value = "Urban";
      document.getElementById("s_avg_glucose_level").value = 240.55;
      document.getElementById("s_bmi").value = 38.2;
      document.getElementById("s_smoking_status").value = "formerly smoked";
    } else {
      document.getElementById("s_gender").value = "Female";
      document.getElementById("s_age").value = 32.0;
      document.getElementById("s_hypertension").value = 0;
      document.getElementById("s_heart_disease").value = 0;
      document.getElementById("s_ever_married").value = "Yes";
      document.getElementById("s_work_type").value = "Private";
      document.getElementById("s_Residence_type").value = "Rural";
      document.getElementById("s_avg_glucose_level").value = 85.10;
      document.getElementById("s_bmi").value = 24.5;
      document.getElementById("s_smoking_status").value = "never smoked";
    }
  }
}

// --- Setup Form Event Listeners ---
function setupFormListeners() {
  const heartForm = document.getElementById("heartForm");
  const strokeForm = document.getElementById("strokeForm");

  heartForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await handleSubmit("heart", heartForm, "btnAnalyzeHeart");
  });

  strokeForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await handleSubmit("stroke", strokeForm, "btnAnalyzeStroke");
  });
}

// --- Form Submission Handler ---
async function handleSubmit(module, formElement, buttonId) {
  const btn = document.getElementById(buttonId);
  const btnText = btn.querySelector(".btn-text");
  const btnSpinner = btn.querySelector(".btn-spinner");
  
  // Set Loading UI State
  btn.disabled = true;
  btnText.textContent = "Analyzing clinical data...";
  btnSpinner.classList.remove("hidden");
  
  hideError();

  try {
    let payload = {};
    let endpoint = "";

    if (module === "heart") {
      endpoint = `${API_BASE_URL}/predict/heart`;
      payload = {
        age: parseInt(document.getElementById("h_age").value, 10),
        sex: parseInt(document.getElementById("h_sex").value, 10),
        cp: parseInt(document.getElementById("h_cp").value, 10),
        trestbps: parseInt(document.getElementById("h_trestbps").value, 10),
        chol: parseInt(document.getElementById("h_chol").value, 10),
        fbs: parseInt(document.getElementById("h_fbs").value, 10),
        restecg: parseInt(document.getElementById("h_restecg").value, 10),
        thalach: parseInt(document.getElementById("h_thalach").value, 10),
        exang: parseInt(document.getElementById("h_exang").value, 10),
        oldpeak: parseFloat(document.getElementById("h_oldpeak").value),
        slope: parseInt(document.getElementById("h_slope").value, 10),
        ca: parseInt(document.getElementById("h_ca").value, 10),
        thal: parseInt(document.getElementById("h_thal").value, 10)
      };
    } else {
      endpoint = `${API_BASE_URL}/predict/stroke`;
      const bmiVal = document.getElementById("s_bmi").value;
      payload = {
        gender: document.getElementById("s_gender").value,
        age: parseFloat(document.getElementById("s_age").value),
        hypertension: parseInt(document.getElementById("s_hypertension").value, 10),
        heart_disease: parseInt(document.getElementById("s_heart_disease").value, 10),
        ever_married: document.getElementById("s_ever_married").value,
        work_type: document.getElementById("s_work_type").value,
        Residence_type: document.getElementById("s_Residence_type").value,
        avg_glucose_level: parseFloat(document.getElementById("s_avg_glucose_level").value),
        bmi: bmiVal !== "" ? parseFloat(bmiVal) : null,
        smoking_status: document.getElementById("s_smoking_status").value
      };
    }

    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || errData.details?.join(", ") || `Server returned status ${response.status}`);
    }

    const data = await response.json();
    renderResult(data);
    
  } catch (error) {
    showError("Clinical Service Notice", error.message || "Failed to connect to the FastAPI clinical server. Ensure the backend is running at http://127.0.0.1:8000.");
  } finally {
    btn.disabled = false;
    btnText.textContent = module === "heart" ? "Run Heart AI Screening" : "Run Stroke AI Screening";
    btnSpinner.classList.add("hidden");
  }
}

// --- Render Result Card ---
function renderResult(data) {
  document.getElementById("placeholderState").classList.add("hidden");
  document.getElementById("errorAlert").classList.add("hidden");
  document.getElementById("resultContent").classList.remove("hidden");

  const isPositive = data.prediction === 1;
  const banner = document.getElementById("resultBanner");
  const moduleChip = document.getElementById("moduleChip");
  const resultStatusText = document.getElementById("resultStatusText");
  const resultMessageText = document.getElementById("resultMessageText");
  const scoreValue = document.getElementById("scoreValue");
  const gaugeFill = document.getElementById("gaugeFill");
  const featureList = document.getElementById("featureList");
  const disclaimerText = document.getElementById("disclaimerText");

  // Module Chip
  moduleChip.textContent = `${data.module.toUpperCase()} MODULE`;

  // Banner State & Text
  if (isPositive) {
    banner.className = "result-banner risk-positive";
    resultStatusText.textContent = "Elevated Risk Pattern Detected";
  } else {
    banner.className = "result-banner risk-negative";
    resultStatusText.textContent = "Baseline Risk Pattern Detected";
  }
  resultMessageText.textContent = data.message;

  // Probability Gauge
  const pct = (data.probability * 100).toFixed(1);
  scoreValue.textContent = `${pct}%`;
  gaugeFill.style.width = `${pct}%`;

  // Render Important Features
  featureList.innerHTML = "";
  if (data.important_features && data.important_features.length > 0) {
    data.important_features.forEach((item) => {
      const isIncrease = item.effect.includes("increases");
      const featureEl = document.createElement("div");
      featureEl.className = "feature-item";
      
      const badgeClass = isIncrease ? "badge-increase" : "badge-lower";
      const arrowIcon = isIncrease ? "↑" : "↓";
      
      featureEl.innerHTML = `
        <div class="feature-main">
          <span class="feature-name">${item.feature}</span>
          <span class="feature-val">Value: ${item.value}</span>
        </div>
        <div class="feature-badge ${badgeClass}">
          <span>${arrowIcon}</span>
          <span>${item.effect}</span>
        </div>
      `;
      featureList.appendChild(featureEl);
    });
  }

  // Safety Disclaimer
  disclaimerText.textContent = data.disclaimer;
}

// --- Error Handling UI ---
function showError(title, message) {
  document.getElementById("placeholderState").classList.add("hidden");
  document.getElementById("resultContent").classList.add("hidden");
  
  const errorAlert = document.getElementById("errorAlert");
  document.getElementById("errorTitle").textContent = title;
  document.getElementById("errorMsg").textContent = message;
  errorAlert.classList.remove("hidden");
}

function hideError() {
  document.getElementById("errorAlert").classList.add("hidden");
}
