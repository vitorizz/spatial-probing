/* Review-session validation is shared by browser restore and offline Node checks. */
(() => {
  "use strict";
  const verdicts = ["unreviewed", "accept", "reject", "unsure"];

  function emptyReview(id) {
    return {id, verdict: "unreviewed", corrected_label: "", notes: ""};
  }

  function validateSession(value, sample) {
    if (!value || value.schema_version !== 1 || value.sample_sha256 !== sample.sample_sha256) {
      throw new Error("This review file belongs to a different sample or format.");
    }
    if (typeof value.reviewer !== "string" || !Array.isArray(value.reviews)) {
      throw new Error("Invalid reviewer or review list.");
    }
    const ids = new Set(sample.items.map(item => item.id));
    const seen = new Set();
    const reviews = value.reviews.map(review => {
      if (!review || !ids.has(review.id) || seen.has(review.id) || !verdicts.includes(review.verdict)
          || typeof review.notes !== "string" || typeof review.corrected_label !== "string"
          || (review.corrected_label !== "" && !sample.options.includes(review.corrected_label))
          || (review.corrected_label !== "" && review.verdict !== "reject")) {
        throw new Error("Invalid review, duplicate question ID, or correction without rejection.");
      }
      seen.add(review.id);
      return {id: review.id, verdict: review.verdict, corrected_label: review.corrected_label, notes: review.notes};
    });
    return {schema_version: 1, sample_sha256: sample.sample_sha256, reviewer: value.reviewer, reviews};
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {emptyReview, validateSession};
    return;
  }

  const sample = JSON.parse(document.getElementById("sample-data").textContent);
  const cards = [...document.querySelectorAll(".card")];
  const reviewer = document.getElementById("reviewer");
  const status = document.getElementById("status-filter");
  const search = document.getElementById("search");
  const message = document.getElementById("message");
  const storageKey = `spatial-handcheck:v1:${sample.sample_sha256}`;
  let reviews = new Map(sample.items.map(item => [item.id, emptyReview(item.id)]));

  function session() {
    return {schema_version: 1, sample_sha256: sample.sample_sha256, reviewer: reviewer.value.trim(),
            reviews: [...reviews.values()]};
  }

  function refresh() {
    const counts = {unreviewed: 0, accept: 0, reject: 0, unsure: 0};
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    cards.forEach((card, index) => {
      const item = sample.items[index];
      const review = reviews.get(item.id);
      counts[review.verdict]++;
      card.dataset.verdict = review.verdict;
      card.hidden = (status.value !== "all" && review.verdict !== status.value)
        || !`${item.id} ${item.question} ${item.label}`.toLowerCase().includes(query);
      card.querySelector(".corrected-label").disabled = review.verdict !== "reject";
      if (!card.hidden) visible++;
    });
    document.getElementById("progress").textContent = `${sample.items.length - counts.unreviewed}/${sample.items.length} reviewed · ${counts.accept} accepted · ${counts.reject} rejected · ${counts.unsure} unsure`;
    document.getElementById("empty").hidden = visible !== 0;
  }

  function save() {
    refresh();
    try {
      localStorage.setItem(storageKey, JSON.stringify(session()));
      message.textContent = "Saved in this browser. Export a JSON copy to share your work.";
    } catch (_) {
      message.textContent = "Browser storage is unavailable. Export a JSON copy to save your work.";
    }
  }

  function restore(value) {
    const checked = validateSession(value, sample);
    reviews = new Map(sample.items.map(item => [item.id, emptyReview(item.id)]));
    checked.reviews.forEach(review => reviews.set(review.id, review));
    reviewer.value = checked.reviewer;
    cards.forEach((card, index) => {
      const review = reviews.get(sample.items[index].id);
      card.querySelector(".verdict").value = review.verdict;
      card.querySelector(".corrected-label").value = review.corrected_label;
      card.querySelector(".notes").value = review.notes;
    });
    refresh();
  }

  cards.forEach((card, index) => card.addEventListener("input", () => {
    const verdict = card.querySelector(".verdict").value;
    const correction = card.querySelector(".corrected-label");
    if (verdict !== "reject") correction.value = "";
    const id = sample.items[index].id;
    reviews.set(id, {id, verdict, corrected_label: correction.value, notes: card.querySelector(".notes").value});
    save();
  }));
  reviewer.addEventListener("input", save);
  search.addEventListener("input", refresh);
  status.addEventListener("change", refresh);

  document.getElementById("export").addEventListener("click", () => {
    if (!reviewer.value.trim()) {
      message.textContent = "Enter your reviewer name before exporting.";
      reviewer.focus();
      return;
    }
    const blob = new Blob([JSON.stringify(session(), null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `reviews_${reviewer.value.trim().replace(/[^a-z0-9_-]/gi, "_")}_${sample.sample_sha256.slice(0, 12)}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    message.textContent = "Review JSON exported. Keep this file to restore or share your decisions.";
  });
  document.getElementById("import").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      restore(JSON.parse(await file.text()));
      save();
      message.textContent = "Review session restored.";
    } catch (error) {
      message.textContent = `Could not restore: ${error.message}`;
    }
    event.target.value = "";
  });
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) restore(JSON.parse(saved));
  } catch (_) {
    message.textContent = "Saved browser session could not be loaded. Restore an exported JSON if available.";
  }
  refresh();
})();
