const form = document.getElementById("predict-form");
const imageInput = document.getElementById("image-input");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("preview-image");
const statusBox = document.getElementById("status");
const result = document.getElementById("result");
const predictedClassEl = document.getElementById("predicted-class");
const confidenceEl = document.getElementById("confidence");
const probabilityList = document.getElementById("probability-list");
const predictBtn = document.getElementById("predict-btn");
const dropzoneTitle = document.getElementById("dropzone-title");

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    previewWrap.hidden = true;
    dropzoneTitle.textContent = "Drop image here or click to browse";
    return;
  }

  dropzoneTitle.textContent = file.name;
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewWrap.hidden = false;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    setStatus("Please choose an image first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  setStatus("Running prediction...", false);
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    renderResult(data);
    setStatus(`Prediction complete for ${data.image_name}.`, false);
  } catch (error) {
    setStatus(error.message || "Something went wrong.", true);
    result.hidden = true;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict Disease Class";
  }
});

function setStatus(message, isError) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function renderResult(data) {
  predictedClassEl.textContent = data.predicted_class;
  confidenceEl.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;

  probabilityList.innerHTML = "";
  data.probabilities.forEach((item) => {
    const row = document.createElement("div");
    row.className = "probability-item";
    row.innerHTML = `<strong>${item.class_name}</strong><span>${item.probability.toFixed(2)}%</span>`;
    probabilityList.appendChild(row);
  });

  result.hidden = false;
}
