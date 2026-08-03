const state = {
  episode: null,
  activeWord: null,
  previewAudio: null,
  taskBusy: false,
  events: null,
  selectedWords: new Set(),
  effectGroups: [],
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
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function getWords() {
  return (state.episode?.sentences || []).flatMap((sentence) => sentence.words || []);
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function renderEpisode() {
  const episode = state.episode;
  const words = getWords();
  el("episodeTitle").textContent = episode ? (episode.display_name || episode.name) : "No audio uploaded";
  if (episode?.transcript) {
    el("episodeMeta").textContent = words.length
      ? `${episode.sentences.length} sentences · ${words.length} aligned words`
      : `${episode.sentences.length} sentences · no word timestamps found`;
  } else {
    el("episodeMeta").textContent = "Transcript not loaded.";
  }

  el("alignmentMeta").textContent = !episode?.transcript
    ? ""
    : `${words.length} timed units available for exact audio preview`;

  const hasEpisode = Boolean(episode?.original);
  el("transcribeBtn").disabled = state.taskBusy || !hasEpisode;
  el("audioRow").hidden = !hasEpisode;
  el("originalAudio").src = mediaUrl(episode?.original);
  const editedUrl = episode?.edited
    ? `${mediaUrl(episode.edited)}?v=${episode.edited_version || ""}`
    : "";
  el("editedAudioWrap").hidden = !editedUrl;
  el("editedAudio").src = editedUrl;
  el("downloadEdited").href = editedUrl;
  el("editToolbar").hidden = !episode?.transcript;
  updateEditControls();
  renderEffectGroups();
  renderSentences();
}

function updateEditControls() {
  const count = state.selectedWords.size;
  el("selectedWordCount").textContent = `${count} word${count === 1 ? "" : "s"} selected`;
  el("addEffectBtn").disabled = state.taskBusy || count === 0;
  el("slowWordsBtn").disabled = state.taskBusy || state.effectGroups.length === 0;
  ["slowSpeed", "gainDb", "pauseBefore", "pauseAfter"].forEach((id) => {
    el(id).disabled = state.taskBusy;
  });
}

function renderEffectGroups() {
  const container = el("effectGroups");
  container.innerHTML = "";
  container.hidden = state.effectGroups.length === 0;
  state.effectGroups.forEach((group, index) => {
    const row = document.createElement("div");
    row.className = "effect-group";
    const words = getWords().filter((word) => group.word_ids.includes(word.index)).map((word) => word.text);
    const summary = document.createElement("span");
    summary.textContent = `Group ${index + 1}: ${words.join(" ")} · ${group.speed.toFixed(2)}× · +${group.gain_db.toFixed(1)} dB · ${group.pause_before_ms}/${group.pause_after_ms} ms`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "compact secondary";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      state.effectGroups.splice(index, 1);
      renderEffectGroups();
      updateEditControls();
    });
    row.append(summary, remove);
    container.appendChild(row);
  });
}

function playWord(word) {
  stopPreview();
  state.activeWord = word.index;
  state.previewAudio = new Audio(mediaUrl(word.word_audio));
  state.previewAudio.addEventListener("ended", stopPreview, { once: true });
  renderSentences();
  state.previewAudio.play()
    .catch((error) => {
      stopPreview();
      log(`Could not play preview: ${error.message}`);
    });
}

function stopPreview() {
  if (state.previewAudio) {
    state.previewAudio.pause();
    state.previewAudio = null;
  }
  state.activeWord = null;
  renderSentences();
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
    const row = document.createElement("article");
    row.className = "sentence-row";
    const header = document.createElement("header");
    header.className = "sentence-header";
    const identity = document.createElement("div");
    const number = document.createElement("span");
    number.className = "sentence-number";
    number.textContent = `Sentence ${String(sentence.index).padStart(2, "0")}`;
    const timing = document.createElement("span");
    timing.className = "time";
    timing.textContent = `${formatTime(sentence.start)}–${formatTime(sentence.end)}`;
    identity.append(number, timing);
    header.appendChild(identity);

    if (sentence.sentence_audio) {
      const preview = document.createElement("audio");
      preview.controls = true;
      preview.preload = "none";
      preview.src = mediaUrl(sentence.sentence_audio);
      preview.setAttribute("aria-label", `Play sentence ${sentence.index}`);
      header.appendChild(preview);
    }

    const wordList = document.createElement("div");
    wordList.className = "word-list";
    if (!sentence.words?.length) {
      const unavailable = document.createElement("p");
      unavailable.className = "untimed-sentence";
      unavailable.textContent = sentence.text || "No timed words in this sentence.";
      wordList.appendChild(unavailable);
    } else {
      sentence.words.forEach((word) => {
        const active = state.activeWord === word.index;
        const chosen = state.selectedWords.has(word.index);
        const item = document.createElement("div");
        item.className = `word-item${chosen ? " chosen" : ""}`;
        const selector = document.createElement("input");
        selector.type = "checkbox";
        selector.checked = chosen;
        selector.setAttribute("aria-label", `Select ${word.text} for slowdown`);
        selector.addEventListener("change", () => {
          if (selector.checked) state.selectedWords.add(word.index);
          else state.selectedWords.delete(word.index);
          item.classList.toggle("chosen", selector.checked);
          updateEditControls();
        });
        const chip = document.createElement("button");
        chip.type = "button";
        chip.id = `word-${word.index}`;
        chip.className = [
          "word-chip",
          active ? "selected" : "",
        ].filter(Boolean).join(" ");
        chip.title = `${formatTime(word.start)}–${formatTime(word.end)}`;

        const text = document.createElement("span");
        text.className = "word-text";
        text.textContent = word.text;
        const interval = document.createElement("span");
        interval.className = "word-meta";
        interval.textContent = `${word.start.toFixed(3)}–${word.end.toFixed(3)}s`;
        const details = document.createElement("span");
        details.className = "word-meta";
        details.textContent = `${(word.end - word.start).toFixed(3)}s duration`;
        const playLabel = document.createElement("span");
        playLabel.className = "word-play";
        playLabel.textContent = active ? "Playing" : "Play";
        chip.append(text, interval, details, playLabel);
        chip.addEventListener("click", () => playWord(word));
        item.append(selector, chip);
        wordList.appendChild(item);
      });
    }
    row.append(header, wordList);
    container.appendChild(row);
  });
}

function watchJob(job) {
  if (state.events) state.events.close();
  setStatus(job.action === "upload-wav" ? "Uploading" : "Running");
  log(`Job ${job.id}: ${job.action}`);
  state.events = new EventSource(`/api/jobs/${job.id}/events`);
  state.events.onmessage = async (event) => {
    if (event.data === "[[DONE]]") {
      state.events.close();
      const finalJob = await api(`/api/jobs/${job.id}`);
      setStatus(finalJob.status === "done" ? "Done" : "Failed");
      if (finalJob.error) log(finalJob.error);
      if (finalJob.action === "upload-wav") {
        state.selectedWords.clear();
        state.effectGroups = [];
        el("uploadHint").textContent = finalJob.status === "done"
          ? `${finalJob.result.display_name || finalJob.result.name} uploaded. Transcribe it when you are ready.`
          : "Upload failed. Choose the WAV again to retry.";
      }
      if (finalJob.action === "transcribe" && finalJob.status === "done") {
        state.selectedWords.clear();
        state.effectGroups = [];
      }
      if (finalJob.result?.name) state.episode = finalJob.result;
      setTaskBusy(false);
      renderEpisode();
      return;
    }
    log(event.data);
  };
  state.events.onerror = () => {
    if (state.events?.readyState !== EventSource.CLOSED) log("Lost the live job log connection.");
  };
}

async function startJob(path, payload) {
  setTaskBusy(true);
  try {
    watchJob(await api(path, payload));
  } catch (error) {
    setTaskBusy(false);
    renderEpisode();
    throw error;
  }
}

function setTaskBusy(busy) {
  state.taskBusy = busy;
  el("chooseFileBtn").disabled = busy;
  el("wavFile").disabled = busy;
  el("uploadPanel").classList.toggle("busy", busy);
  if (busy) el("transcribeBtn").disabled = true;
  updateEditControls();
}

async function uploadWav(file) {
  if (!file || !file.name.toLowerCase().endsWith(".wav")) {
    el("uploadHint").textContent = "Please choose a WAV file.";
    log("Upload rejected: only WAV files are supported.");
    return;
  }
  setTaskBusy(true);
  setStatus("Uploading");
  el("uploadHint").textContent = `Uploading ${file.name}…`;
  try {
    const response = await fetch("/api/upload-wav", {
      method: "POST",
      headers: { "Content-Type": file.type || "audio/wav", "X-Filename": encodeURIComponent(file.name) },
      body: file,
    });
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || response.statusText);
    watchJob(job);
  } catch (error) {
    setTaskBusy(false);
    renderEpisode();
    setStatus("Failed");
    el("uploadHint").textContent = "Upload failed. Choose the WAV again to retry.";
    log(error.message);
  }
}

el("chooseFileBtn").addEventListener("click", () => { if (!state.taskBusy) el("wavFile").click(); });
el("wavFile").addEventListener("change", (event) => {
  const [file] = event.target.files;
  event.target.value = "";
  uploadWav(file);
});
["dragenter", "dragover"].forEach((eventName) => el("uploadPanel").addEventListener(eventName, (event) => {
  event.preventDefault();
  if (!state.taskBusy) el("uploadPanel").classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => el("uploadPanel").addEventListener(eventName, (event) => {
  event.preventDefault();
  el("uploadPanel").classList.remove("dragging");
}));
el("uploadPanel").addEventListener("drop", (event) => {
  if (!state.taskBusy) uploadWav(event.dataTransfer.files[0]);
});
el("transcribeBtn").addEventListener("click", () => {
  startJob("/api/transcribe", { episode: state.episode.name }).catch((error) => log(error.message));
});
el("slowWordsBtn").addEventListener("click", () => {
  startJob("/api/slow-words", {
    episode: state.episode.name,
    edits: state.effectGroups,
  }).catch((error) => log(error.message));
});
el("addEffectBtn").addEventListener("click", () => {
  const selected = [...state.selectedWords].sort((a, b) => a - b);
  const assigned = new Set(state.effectGroups.flatMap((group) => group.word_ids));
  const overlap = selected.filter((wordId) => assigned.has(wordId));
  if (overlap.length) {
    log(`Words already assigned to another effect group: ${overlap.join(", ")}`);
    return;
  }
  state.effectGroups.push({
    word_ids: selected,
    speed: Number(el("slowSpeed").value),
    gain_db: Number(el("gainDb").value),
    pause_before_ms: Number(el("pauseBefore").value),
    pause_after_ms: Number(el("pauseAfter").value),
  });
  state.selectedWords.clear();
  renderEpisode();
});
el("clearLogBtn").addEventListener("click", () => { el("log").textContent = ""; });

api("/api/current")
  .then((data) => { state.episode = data.episode; renderEpisode(); })
  .catch((error) => log(error.message));
