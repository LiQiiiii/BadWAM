const copyButton = document.querySelector("[data-copy-bibtex]");
const copyStatus = document.querySelector("[data-copy-status]");
const bibtex = document.querySelector("#bibtex");

if (copyButton && copyStatus && bibtex) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      copyStatus.textContent = "Copied.";
      window.setTimeout(() => {
        copyStatus.textContent = "";
      }, 1800);
    } catch (error) {
      copyStatus.textContent = "Copy failed. Please select the text manually.";
    }
  });
}

const navLinks = [...document.querySelectorAll(".site-nav a")];
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  },
  { rootMargin: "-22% 0px -68% 0px", threshold: [0.1, 0.3, 0.6] }
);

sections.forEach((section) => observer.observe(section));

