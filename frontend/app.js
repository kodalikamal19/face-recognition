// ==========================================================================
// Face Recognition Frontend Application Logic
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const dropZonePrompt = document.getElementById("dropZonePrompt");
  const previewWrapper = document.getElementById("previewWrapper");
  const imagePreview = document.getElementById("imagePreview");
  const previewFilename = document.getElementById("previewFilename");
  const removeImageBtn = document.getElementById("removeImageBtn");
  const scanButton = document.getElementById("scanButton");
  const sampleChips = document.getElementById("sampleChips");

  const systemStatusBadge = document.getElementById("systemStatusBadge");
  const systemStatusText = document.getElementById("systemStatusText");

  const processingSection = document.getElementById("processingSection");
  const processingStatus = document.getElementById("processingStatus");
  const resultsSection = document.getElementById("resultsSection");

  const statRollCount = document.getElementById("statRollCount");
  const statTotalFaces = document.getElementById("statTotalFaces");
  const statUnknownFaces = document.getElementById("statUnknownFaces");
  const statProcessTime = document.getElementById("statProcessTime");

  const rollNumbersGrid = document.getElementById("rollNumbersGrid");
  const emptyRollsState = document.getElementById("emptyRollsState");
  const rollNumbersSubtitle = document.getElementById("rollNumbersSubtitle");
  const copyRollsBtn = document.getElementById("copyRollsBtn");
  const downloadCsvBtn = document.getElementById("downloadCsvBtn");

  const toggleVisualBtn = document.getElementById("toggleVisualBtn");
  const visualCard = document.querySelector(".visual-card");
  const annotatedImage = document.getElementById("annotatedImage");
  const downloadImageBtn = document.getElementById("downloadImageBtn");

  const toast = document.getElementById("toast");
  const toastMessage = document.getElementById("toastMessage");

  // State
  let currentFile = null;
  let currentSampleFilename = null;
  let latestResult = null;
  let toastTimer = null;

  // 1. Initial Status Check
  async function checkSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("Status API returned error");
      const data = await res.json();
      systemStatusText.textContent = `${data.enrolled_count} Students Enrolled`;
      systemStatusBadge.classList.add("ready");
    } catch (err) {
      console.warn("Status check:", err);
      systemStatusText.textContent = "Backend Offline";
      systemStatusBadge.style.color = "#dc2626";
    }
  }

  // 2. Load Sample Images
  async function loadSampleImages() {
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) throw new Error("Could not load samples");
      const data = await res.json();

      sampleChips.innerHTML = "";
      if (!data.samples || data.samples.length === 0) {
        sampleChips.innerHTML = '<span class="sample-label">No test samples found</span>';
        return;
      }

      data.samples.slice(0, 5).forEach((sample) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "sample-chip";
        chip.textContent = sample.filename;
        chip.addEventListener("click", () => selectSample(sample.filename, chip));
        sampleChips.appendChild(chip);
      });
    } catch (err) {
      console.warn("Samples load error:", err);
      sampleChips.innerHTML = '<span class="sample-label">Upload a photo to start</span>';
    }
  }

  // Select Sample Image
  function selectSample(filename, chipElement) {
    document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));
    chipElement.classList.add("active");

    currentSampleFilename = filename;
    currentFile = null;
    fileInput.value = "";

    const imageUrl = `/api/sample/${encodeURIComponent(filename)}`;
    imagePreview.src = imageUrl;
    previewFilename.textContent = filename;

    dropZonePrompt.classList.add("hidden");
    previewWrapper.classList.remove("hidden");
    scanButton.disabled = false;
  }

  // 3. Drag and Drop Handling
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("drag-active");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      showToast("Please upload an image file (JPG, PNG, WEBP)");
      return;
    }

    currentFile = file;
    currentSampleFilename = null;
    document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewFilename.textContent = file.name;
      dropZonePrompt.classList.add("hidden");
      previewWrapper.classList.remove("hidden");
      scanButton.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Remove Selected Image
  removeImageBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUploadState();
  });

  function resetUploadState() {
    currentFile = null;
    currentSampleFilename = null;
    fileInput.value = "";
    imagePreview.src = "";
    previewWrapper.classList.add("hidden");
    dropZonePrompt.classList.remove("hidden");
    scanButton.disabled = true;
    document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));
  }

  // 4. Scanning & Recognition Trigger
  scanButton.addEventListener("click", async () => {
    if (!currentFile && !currentSampleFilename) return;

    // Show processing state
    scanButton.disabled = true;
    resultsSection.classList.add("hidden");
    processingSection.classList.remove("hidden");
    processingSection.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Animated status messages during detection
    const statuses = [
      "Detecting multi-faces with MTCNN...",
      "Aligning 5-point facial landmarks...",
      "Extracting 512-d FaceNet embeddings...",
      "Running Cosine similarity identity matching...",
      "Finalizing roll number records..."
    ];
    let statusIdx = 0;
    const statusInterval = setInterval(() => {
      statusIdx = (statusIdx + 1) % statuses.length;
      processingStatus.textContent = statuses[statusIdx];
    }, 700);

    try {
      let response;
      if (currentFile) {
        const formData = new FormData();
        formData.append("file", currentFile);
        response = await fetch("/api/recognize", {
          method: "POST",
          body: formData
        });
      } else {
        response = await fetch(`/api/recognize-sample?filename=${encodeURIComponent(currentSampleFilename)}`, {
          method: "POST"
        });
      }

      clearInterval(statusInterval);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Face recognition failed");
      }

      const data = await response.json();
      latestResult = data;
      renderResults(data);
    } catch (err) {
      clearInterval(statusInterval);
      console.error(err);
      showToast(err.message || "An error occurred while recognizing faces");
    } finally {
      processingSection.classList.add("hidden");
      scanButton.disabled = false;
    }
  });

  // 5. Render Results Function
  function renderResults(data) {
    // 1. Stats Bar
    statRollCount.textContent = data.recognized_count;
    statTotalFaces.textContent = data.total_faces_detected;
    statUnknownFaces.textContent = data.unknown_count;
    statProcessTime.textContent = `${data.processing_time_seconds}s`;

    // 2. Roll Numbers Grid
    rollNumbersGrid.innerHTML = "";
    const rollNumbers = data.roll_numbers || [];

    if (rollNumbers.length === 0) {
      emptyRollsState.classList.remove("hidden");
      rollNumbersSubtitle.textContent = "No enrolled roll numbers found in this image";
    } else {
      emptyRollsState.classList.add("hidden");
      rollNumbersSubtitle.textContent = `${rollNumbers.length} enrolled student${rollNumbers.length > 1 ? "s" : ""} recognized`;

      // Map roll numbers to their highest similarity score
      const rollScores = {};
      (data.faces || []).forEach((f) => {
        if (f.is_recognized && f.roll_number !== "UNKNOWN") {
          const currentMax = rollScores[f.roll_number] || 0;
          if (f.similarity_percent > currentMax) {
            rollScores[f.roll_number] = f.similarity_percent;
          }
        }
      });

      rollNumbers.forEach((roll) => {
        const card = document.createElement("div");
        card.className = "roll-card";
        card.title = `Click to copy ${roll}`;

        const score = rollScores[roll] || 85;

        card.innerHTML = `
          <div class="roll-card-info">
            <span class="roll-dot"></span>
            <span class="roll-number-text">${roll}</span>
          </div>
          <div class="roll-meta">
            <span class="confidence-tag">${score}%</span>
            <svg class="copy-mini-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </div>
        `;

        card.addEventListener("click", () => {
          navigator.clipboard.writeText(roll).then(() => {
            showToast(`Copied ${roll}`);
          });
        });

        rollNumbersGrid.appendChild(card);
      });
    }

    // 3. Visual Preview Image
    if (data.annotated_image) {
      annotatedImage.src = data.annotated_image;
      downloadImageBtn.href = data.annotated_image;
      downloadImageBtn.download = `recognized_${data.filename || "output"}.jpg`;
    }

    // Show Results
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // 6. Copy All Roll Numbers
  copyRollsBtn.addEventListener("click", () => {
    if (!latestResult || !latestResult.roll_numbers || latestResult.roll_numbers.length === 0) {
      showToast("No roll numbers to copy");
      return;
    }
    const textToCopy = latestResult.roll_numbers.join(", ");
    navigator.clipboard.writeText(textToCopy).then(() => {
      showToast(`Copied ${latestResult.roll_numbers.length} roll numbers!`);
    });
  });

  // 7. Download CSV
  downloadCsvBtn.addEventListener("click", () => {
    if (!latestResult || !latestResult.roll_numbers || latestResult.roll_numbers.length === 0) {
      showToast("No roll numbers to export");
      return;
    }
    let csvContent = "data:text/csv;charset=utf-8,Roll Number\n";
    latestResult.roll_numbers.forEach((roll) => {
      csvContent += `${roll}\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `roll_numbers_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("CSV downloaded");
  });

  // 8. Toggle Visual Inspection
  toggleVisualBtn.addEventListener("click", () => {
    visualCard.classList.toggle("collapsed");
  });

  // 9. Toast Notification Helper
  function showToast(msg) {
    toastMessage.textContent = msg;
    toast.classList.add("show");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove("show");
    }, 2400);
  }

  // Initialize
  checkSystemStatus();
  loadSampleImages();
});
