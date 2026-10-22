// Copy-to-clipboard for the BibTeX block.
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("copy-bibtex");
  const code = document.getElementById("bibtex");
  if (!btn || !code) return;
  btn.addEventListener("click", async () => {
    const label = btn.querySelector("span:last-child");
    try {
      await navigator.clipboard.writeText(code.innerText);
      label.textContent = "Copied!";
    } catch (e) {
      label.textContent = "Copy failed";
    }
    setTimeout(() => { label.textContent = "Copy"; }, 1500);
  });
});
