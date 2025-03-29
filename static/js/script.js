document.addEventListener("DOMContentLoaded", function () {
  const registerSection = document.getElementById("registerSection");
  const checkSection = document.getElementById("checkSection");
  const registerUserBtn = document.getElementById("registerUser");
  const checkFaceBtn = document.getElementById("checkFace");

  registerUserBtn.addEventListener("click", () => {
    registerSection.classList.remove("hidden");
    checkSection.classList.add("hidden");
  });

  checkFaceBtn.addEventListener("click", () => {
    checkSection.classList.remove("hidden");
    registerSection.classList.add("hidden");
  });

  document.getElementById("registerBtn").addEventListener("click", async () => {
    const file = document.getElementById("uploadImage").files[0];
    if (!file) {
      alert("Please upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    document.getElementById("registerMessage").innerText = result.message;
  });

  document.getElementById("checkBtn").addEventListener("click", async () => {
    const file = document.getElementById("checkImage").files[0];
    if (!file) {
      alert("Please upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch("/recognize", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    document.getElementById("checkMessage").innerText = result.message;
  });
});
