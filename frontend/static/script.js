function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Upload an image!");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    // 🔥 Show loader
    document.getElementById("loader").classList.remove("hidden");

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {

        // 🔥 Hide loader
        document.getElementById("loader").classList.add("hidden");

        // =====================================
        // 🧠 RESULT DISPLAY
        // =====================================
        let resultDiv = document.getElementById("result");
        resultDiv.classList.remove("hidden");

        let statusColor = data.result === "normal" ? "lime" : "red";

        resultDiv.innerHTML = `
            <h2 style="color:${statusColor}">
                ${data.result.toUpperCase()}
            </h2>
            <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
        `;

        // =====================================
        // 🖼️ SHOW UPLOADED IMAGE
        // =====================================
        let uploadedImg = document.getElementById("uploadedImg");
        if (uploadedImg) {
            uploadedImg.src = data.uploaded;
            uploadedImg.classList.remove("hidden");
        }

        // =====================================
        // 🔥 SHOW HEATMAP
        // =====================================
        let heatmap = document.getElementById("heatmap");
        heatmap.src = data.heatmap + "?t=" + new Date().getTime(); // 🔥 prevent cache
        heatmap.classList.remove("hidden");

    })
    .catch(err => {
        console.error(err);
        document.getElementById("loader").classList.add("hidden");
        alert("Error processing image!");
    });
}