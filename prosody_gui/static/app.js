const state = {
  episodes: [],
  episode: null,
  selected: new Set(),
  events: null,
};

const el = (id) => document.getElementById(id);

function mediaUrl(path) {
  return path ? `/media/${encodeURI(path)}` : "";
}

function setStatus(text) {
  el("status").textContent = text;
}

function log(message) {
  const box = el("log");
  box.textContent += `${message}\n`;
  box.scrollTop = box.scrollHeight;
}

async function api(path, body = null) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

async function refreshEpisodes() {
  const data = await api("/api/episodes");
  state.episodes = data.episodes;
  renderEpisodes();
  renderSettings(data.config);
}

function renderEpisodes() {
  const list = el("episodeList");
  list.innerHTML = "";
  if (!state.episodes.length) {
    list.innerHTML = '<span class="badge">No episodes yet</span>';
    return;
  }
  state.episodes.forEach((episode) => {
    const button = document.createElement("button");
    button.className = `episode-item ${state.episode?.name === episode.name ? "active" : ""}`;
    button.textContent = episode.name;
    button.addEventListener("click", () => loadEpisode(episode.name));
    list.appendChild(button);
  });
}

function renderSettings(config) {
  el("inputDir").value = config.input_dir || "";
  el("outputDir").value = config.output_dir || "";
  const settings = el("settings");
  settings.innerHTML = "";
  Object.entries(config).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.innerHTML = `<dt>${key}</dt><dd>${value}</dd>`;
    settings.appendChild(row);
  });
}

async function loadEpisode(name) {
  const episode = await api(`/api/episode/${encodeURIComponent(name)}`);
  state.episode = episode;
  state.selected.clear();
  renderEpisodes();
  renderEpisode();
}

function renderEpisode() {
  const episode = state.episode;
  const selectedSentences = getSelectedSentences();
  const hasSelection = selectedSentences.length > 0;
  const selectedGenerated = selectedSentences.filter((sentence) => sentence.generated_audio || sentence.trimmed_audio);
  const allSelectedGenerated = hasSelection && selectedGenerated.length === selectedSentences.length;
  el("episodeTitle").textContent = episode ? episode.name : "No episode selected";
  el("episodeMeta").textContent = episode?.transcript
    ? `${episode.sentences.length} sentences loaded`
    : "Transcript not loaded.";
  renderSelectionMeta(selectedSentences, allSelectedGenerated);

  const hasEpisode = Boolean(episode?.original);
  el("transcribeBtn").disabled = !hasEpisode;
  el("generateBtn").disabled = !episode?.sentences?.length || !hasSelection;
  el("spliceBtn").disabled = !episode?.sentences?.length || !allSelectedGenerated;
  el("runAllBtn").disabled = !hasEpisode || !hasSelection;

  el("audioRow").hidden = !hasEpisode;
  el("originalAudio").src = mediaUrl(episode?.original);
  el("finalAudio").src = mediaUrl(episode?.final);
  el("finalLink").textContent = episode?.final || "";
  el("finalLink").href = mediaUrl(episode?.final);

  renderSentences();
}

function getSelectedSentences() {
  const sentences = state.episode?.sentences || [];
  return sentences.filter((sentence) => state.selected.has(sentence.index));
}

function renderSelectionMeta(selectedSentences, allSelectedGenerated) {
  const target = el("selectionMeta");
  target.className = "selection-meta";
  if (!state.episode?.sentences?.length) {
    target.textContent = "";
    return;
  }
  if (!selectedSentences.length) {
    target.textContent = "Select one or more sentences to generate and splice.";
    target.classList.add("warn");
    return;
  }
  const ids = selectedSentences.map((sentence) => String(sentence.index).padStart(2, "0")).join(", ");
  if (!allSelectedGenerated) {
    target.textContent = `Selected: ${ids}. Generate selected sentences before splicing.`;
    target.classList.add("warn");
    return;
  }
  target.textContent = `Selected: ${ids}. Ready to splice.`;
  target.classList.add("ready");
}

function renderSentences() {
  const container = el("sentences");
  container.innerHTML = "";
  const sentences = state.episode?.sentences || [];
  if (!sentences.length) {
    container.className = "sentences empty";
    container.textContent = "No transcript loaded.";
    return;
  }
  container.className = "sentences";
  sentences.forEach((sentence) => {
    const row = document.createElement("div");
    const selected = state.selected.has(sentence.index);
    row.className = `sentence-row ${selected ? "selected" : ""}`;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selected;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.selected.add(sentence.index);
      else state.selected.delete(sentence.index);
      renderEpisode();
    });

    const original = sentence.sentence_audio
      ? `<audio controls src="${mediaUrl(sentence.sentence_audio)}"></audio>`
      : '<span class="badge">not cut</span>';
    const generated = sentence.trimmed_audio || sentence.generated_audio
      ? `<audio controls src="${mediaUrl(sentence.trimmed_audio || sentence.generated_audio)}"></audio>`
      : '<span class="badge">not generated</span>';

    row.appendChild(checkbox);
    row.insertAdjacentHTML("beforeend", `
      <div>${sentence.index}</div>
      <div class="time">${sentence.start.toFixed(3)}-${sentence.end.toFixed(3)}</div>
      <div class="sentence-text">${escapeHtml(sentence.text)}</div>
      <div>${original}</div>
      <div>${generated}</div>
    `);
    container.appendChild(row);
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[ch]));
}

async function startJob(path, payload) {
  if (state.events) state.events.close();
  const job = await api(path, payload);
  setStatus("Running");
  log(`Job ${job.id}: ${job.action}`);
  state.events = new EventSource(`/api/jobs/${job.id}/events`);
  state.events.onmessage = async (event) => {
    if (event.data === "[[DONE]]") {
      state.events.close();
      const finalJob = await api(`/api/jobs/${job.id}`);
      setStatus(finalJob.status === "done" ? "Done" : "Failed");
      if (finalJob.error) log(finalJob.error);
      if (finalJob.result?.name) {
        state.episode = finalJob.result;
        await refreshEpisodes();
        renderEpisode();
      } else if (state.episode?.name) {
        await loadEpisode(state.episode.name);
      }
      return;
    }
    log(event.data);
  };
}

function selectedPayload() {
  if (!state.episode) throw new Error("Choose an episode first.");
  return {
    episode: state.episode.name,
    sentence_ids: Array.from(state.selected).sort((a, b) => a - b),
  };
}

el("importBtn").addEventListener("click", async () => {
  const path = el("wavPath").value.trim();
  if (!path) return log("Enter an absolute WAV path.");
  await startJob("/api/import-wav", { path, name: el("episodeName").value.trim() });
});

el("saveDirsBtn").addEventListener("click", async () => {
  try {
    const data = await api("/api/directories", {
      input_dir: el("inputDir").value.trim(),
      output_dir: el("outputDir").value.trim(),
    });
    state.episodes = data.episodes;
    state.episode = null;
    state.selected.clear();
    renderEpisodes();
    renderEpisode();
    renderSettings(data.config);
    log(`Folders saved. Output: ${data.config.output_dir}`);
  } catch (error) {
    log(error.message);
  }
});

el("transcribeBtn").addEventListener("click", () => {
  startJob("/api/transcribe", { episode: state.episode.name });
});

el("generateBtn").addEventListener("click", () => {
  startJob("/api/generate", selectedPayload()).catch((error) => log(error.message));
});

el("spliceBtn").addEventListener("click", () => {
  startJob("/api/splice", selectedPayload()).catch((error) => log(error.message));
});

el("runAllBtn").addEventListener("click", () => {
  startJob("/api/run-all", selectedPayload()).catch((error) => log(error.message));
});

el("clearLogBtn").addEventListener("click", () => {
  el("log").textContent = "";
});

refreshEpisodes().catch((error) => log(error.message));
