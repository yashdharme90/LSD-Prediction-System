const form = document.querySelector("form");

form.addEventListener("submit", () => {
  document.getElementById("loading").style.display = "block";
});

// Animate confidence bar
document.querySelectorAll(".conf-bar").forEach(bar => {
  const width = bar.getAttribute("data-width");
  if (width) {
    bar.style.width = width + "%";
  }
});