document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll(".hca-btn");
    const hiddenInput = document.getElementById("hca");
  
    buttons.forEach(btn => {
      btn.addEventListener("click", () => {

        buttons.forEach(b => b.classList.remove("active"));

        btn.classList.add("active");

        hiddenInput.value = btn.dataset.value;
      });
    });
  });