const initNav = () => {
  const nav = document.getElementById("nav");
  if (!nav) return;
  const onScroll = () => {
    if (window.scrollY > 8) {
      nav.classList.add("scrolled");
    } else {
      nav.classList.remove("scrolled");
    }
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });
};

const initHeadline = () => {
  const hero = document.querySelector(".hero");
  const headline = document.querySelector("[data-headline]");
  if (!hero || !headline) return;

  const text = headline.textContent.trim();
  headline.textContent = "";
  headline.setAttribute("aria-label", text);
  [...text].forEach((char, index) => {
    const span = document.createElement("span");
    span.className = "headline-char";
    span.style.setProperty("--delay", `${index * 26}ms`);
    span.textContent = char === " " ? "\u00a0" : char;
    span.setAttribute("aria-hidden", "true");
    headline.appendChild(span);
  });
  const caret = document.createElement("span");
  caret.className = "headline-caret";
  caret.setAttribute("aria-hidden", "true");
  headline.appendChild(caret);

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          hero.classList.add("is-revealed");
          observer.disconnect();
        }
      });
    },
    { threshold: 0.4 }
  );

  observer.observe(hero);
};

const initReveals = () => {
  const elements = document.querySelectorAll(".reveal");
  if (!elements.length) return;
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.2, rootMargin: "0px 0px -10% 0px" }
  );

  elements.forEach((el) => observer.observe(el));
};

const initSticky = () => {
  const steps = document.querySelectorAll(".sticky-step");
  const media = document.querySelector(".sticky-media");
  if (!steps.length || !media) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const step = entry.target.getAttribute("data-step");
          media.setAttribute("data-active", step);
          steps.forEach((item) => item.classList.remove("is-active"));
          entry.target.classList.add("is-active");
        }
      });
    },
    { threshold: 0.6 }
  );

  steps.forEach((step) => observer.observe(step));
};

const initStudio = () => {
  const form = document.querySelector("[data-upload-form]");
  const lookupForm = document.querySelector("[data-lookup-form]");
  const dropZone = document.querySelector("[data-drop-zone]");
  const fileName = document.querySelector("[data-file-name]");
  const uploadButton = document.querySelector("[data-upload-button]");
  const uploadNote = document.querySelector("[data-upload-note]");
  const statusPill = document.querySelector("[data-status-pill]");
  const statusJob = document.querySelector("[data-status-job]");
  const statusDetail = document.querySelector("[data-status-detail]");
  const artifactList = document.querySelector("[data-artifact-list]");
  const artifactMeta = document.querySelector("[data-artifact-meta]");
  const apiKeyInput = document.getElementById("apiKey");
  const languageSelect = document.getElementById("language");
  const statusSteps = Array.from(document.querySelectorAll(".status-step"));

  if (!form || !statusPill || !statusJob || !statusDetail || !artifactList || !artifactMeta) {
    return;
  }

  let currentJobId = null;
  let eventSource = null;
  const defaultUploadNote = uploadNote ? uploadNote.textContent : "";
  const allowedLanguages = languageSelect
    ? new Set([...languageSelect.options].map((opt) => opt.value).filter(Boolean))
    : new Set();

  const getApiKey = () => (apiKeyInput ? apiKeyInput.value.trim() : "");
  const setUploadNote = (message, tone) => {
    if (!uploadNote) return;
    uploadNote.textContent = message;
    uploadNote.classList.remove("is-error", "is-success");
    if (tone) {
      uploadNote.classList.add(`is-${tone}`);
    }
  };
  const authHeaders = () => {
    const apiKey = getApiKey();
    return apiKey ? { "x-api-key": apiKey } : {};
  };

  const setStatus = ({ status, stage, detail, error, jobId }) => {
    if (jobId) {
      currentJobId = jobId;
      statusJob.textContent = jobId;
    }
    const normalized = status || "idle";
    const labelMap = {
      queued: "Queued",
      running: "Running",
      succeeded: "Complete",
      failed: "Failed",
      idle: "Idle",
    };
    statusPill.textContent = labelMap[normalized] || normalized;
    statusPill.classList.remove("is-running", "is-succeeded", "is-failed");
    if (normalized === "running") statusPill.classList.add("is-running");
    if (normalized === "succeeded") statusPill.classList.add("is-succeeded");
    if (normalized === "failed") statusPill.classList.add("is-failed");

    statusSteps.forEach((step) => {
      step.classList.remove("is-active", "is-done", "is-error");
    });

    const stepMap = {
      queued: 0,
      running: 1,
      succeeded: 2,
      failed: 2,
    };
    const activeIndex = stepMap[normalized] ?? 0;
    statusSteps.forEach((step, index) => {
      if (index < activeIndex) {
        step.classList.add("is-done");
      } else if (index === activeIndex) {
        step.classList.add(normalized === "failed" ? "is-error" : "is-active");
      }
    });

    let message = detail;
    if (!message && stage === "start") {
      message = "Processing: extracting, drafting, and assembling artifacts.";
    }
    if (!message && stage === "queued") {
      message = "Queued and waiting for the worker.";
    }
    if (!message && normalized === "succeeded") {
      message = "Complete. Artifacts are ready below.";
    }
    if (!message && normalized === "failed") {
      message = "Failed. Check logs for details.";
    }
    if (error) {
      message = `Error: ${error}`;
    }
    statusDetail.textContent = message || "Waiting for input.";
  };

  const setArtifacts = (artifacts) => {
    const apiKey = getApiKey();
    artifactList.innerHTML = "";
    if (!artifacts || !artifacts.length) {
      artifactMeta.textContent = "No artifacts yet";
      return;
    }
    artifactMeta.textContent = `${artifacts.length} files`;
    artifacts.forEach((artifact) => {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = artifact.name;
      const meta = document.createElement("span");
      meta.className = "artifact-kind";
      meta.textContent = artifact.kind || "artifact";
      const link = document.createElement("a");
      const linkUrl = artifact.download_url || "#";
      link.href = apiKey ? `${linkUrl}?api_key=${encodeURIComponent(apiKey)}` : linkUrl;
      link.className = "artifact-link";
      link.textContent = "Download";
      item.append(name, meta, link);
      artifactList.appendChild(item);
    });
  };

  const fetchArtifacts = async (jobId) => {
    try {
      const res = await fetch(`/v1/jobs/${jobId}/artifacts`, {
        headers: authHeaders(),
      });
      if (!res.ok) {
        artifactMeta.textContent = "Artifacts unavailable";
        return;
      }
      const data = await res.json();
      setArtifacts(data.artifacts || []);
    } catch (err) {
      artifactMeta.textContent = "Artifacts unavailable";
    }
  };

  const startProgressStream = (jobId) => {
    if (eventSource) {
      eventSource.close();
    }
    const apiKey = getApiKey();
    const qs = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : "";
    eventSource = new EventSource(`/v1/jobs/${jobId}/progress${qs}`);
    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setStatus({
          status: payload.status,
          stage: payload.stage,
          detail: payload.detail,
          error: payload.error,
          jobId,
        });
        if (payload.status === "succeeded") {
          fetchArtifacts(jobId);
          eventSource.close();
        }
        if (payload.status === "failed") {
          eventSource.close();
        }
      } catch (err) {
        eventSource.close();
      }
    };
  };

  const submitUpload = async (event) => {
    event.preventDefault();
    if (!uploadButton) return;
    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading...";
    setUploadNote(defaultUploadNote, null);

    const fileInput = form.querySelector("input[type='file']");
    if (!fileInput || !fileInput.files || !fileInput.files.length) {
      uploadButton.disabled = false;
      uploadButton.textContent = "Generate episode";
      setUploadNote("Select a PDF to upload.", "error");
      return;
    }
    const languageValue = languageSelect ? languageSelect.value : "";
    if (!languageValue || (allowedLanguages.size && !allowedLanguages.has(languageValue))) {
      uploadButton.disabled = false;
      uploadButton.textContent = "Generate episode";
      setUploadNote("Select a language from the list.", "error");
      return;
    }
    setStatus({ status: "queued", stage: "queued", detail: "Uploading file..." });
    setArtifacts([]);
    try {
      const formData = new FormData(form);
      const response = await fetch("/v1/jobs", {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Upload failed");
      }
      const data = await response.json();
      setStatus({ status: data.status, stage: "queued", jobId: data.job_id });
      setUploadNote("Upload received. Job is running.", "success");
      startProgressStream(data.job_id);
    } catch (err) {
      setStatus({ status: "failed", error: err.message });
      setUploadNote("Upload failed. Check the file and try again.", "error");
    } finally {
      uploadButton.disabled = false;
      uploadButton.textContent = "Generate episode";
    }
  };

  const submitLookup = async (event) => {
    event.preventDefault();
    const input = lookupForm ? lookupForm.querySelector("input[name='job_id']") : null;
    const jobId = input ? input.value.trim() : "";
    if (!jobId) return;
    setStatus({ status: "running", detail: "Loading job status...", jobId });
    try {
      const res = await fetch(`/v1/jobs/${jobId}/status`, { headers: authHeaders() });
      if (!res.ok) throw new Error("Job not found");
      const data = await res.json();
      setStatus({
        status: data.status,
        stage: data.stage,
        detail: data.detail,
        error: data.error,
        jobId,
      });
      startProgressStream(jobId);
      if (data.status === "succeeded") {
        fetchArtifacts(jobId);
      }
    } catch (err) {
      setStatus({ status: "failed", error: err.message, jobId });
    }
  };

  if (dropZone) {
    dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("is-dragover"));
    dropZone.addEventListener("drop", () => dropZone.classList.remove("is-dragover"));
  }

  const fileInput = form.querySelector("input[type='file']");
  if (fileInput && fileName) {
    fileInput.addEventListener("change", () => {
      fileName.textContent = fileInput.files?.[0]?.name || "No file selected";
    });
  }

  form.addEventListener("submit", submitUpload);
  if (lookupForm) {
    lookupForm.addEventListener("submit", submitLookup);
  }
};

const initFeedback = () => {
  const form = document.querySelector("[data-feedback-form]");
  const note = document.querySelector("[data-feedback-note]");
  if (!form || !note) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    note.textContent = "Sending...";
    const formData = new FormData(form);
    const payload = {
      name: formData.get("name") || null,
      email: formData.get("email") || null,
      job_id: formData.get("job_id") || null,
      message: formData.get("message") || "",
      context: formData.get("context")
        ? { note: formData.get("context") }
        : null,
    };
    if (!payload.message.trim()) {
      note.textContent = "Please enter a message.";
      return;
    }
    const apiKeyInput = document.getElementById("apiKey");
    const apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
    try {
      const res = await fetch("/v1/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { "x-api-key": apiKey } : {}),
        },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Feedback failed");
      note.textContent = "Thanks. Feedback received.";
      form.reset();
    } catch (err) {
      note.textContent = "Unable to send feedback.";
    }
  });
};

window.addEventListener("DOMContentLoaded", () => {
  initNav();
  initHeadline();
  initReveals();
  initSticky();
  initStudio();
  initFeedback();
});
