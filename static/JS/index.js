// let selectedImages = {
// 	naivebayes: null,
// 	cnn: null,
// };

// function switchChat(method) {
// 	document
// 		.querySelectorAll(".chat-area")
// 		.forEach((area) => area.classList.remove("active"));
// 	document.getElementById(`chat-${method}`).classList.add("active");
// }

// function setupChat(method) {
// 	const form = document.getElementById(`form-${method}`);
// 	const input = document.getElementById(`input-${method}`);
// 	const fileInput = document.getElementById(`file-${method}`);
// 	const messages = document.getElementById(`messages-${method}`);

// 	fileInput.addEventListener("change", () => {
// 		const file = fileInput.files[0];
// 		if (file && file.type.startsWith("image/")) {
// 			const reader = new FileReader();
// 			reader.onload = function (e) {
// 				selectedImages[method] = e.target.result;
// 			};
// 			reader.readAsDataURL(file);
// 		}
// 	});

// 	form.addEventListener("submit", (e) => {
// 		e.preventDefault();
// 		const message = input.value.trim();
// 		const image = selectedImages[method];

// 		if (!message && !image) return;

// 		const userMsg = document.createElement("div");
// 		userMsg.className = "message user";
// 		let userContent = `<div class="bubble">`;
// 		if (message) userContent += `<p>${message}</p>`;
// 		if (image) userContent += `<img src="${image}" class="chat-image" />`;
// 		userContent += `</div>`;
// 		userMsg.innerHTML = userContent;
// 		messages.appendChild(userMsg);

// 		const botMsg = document.createElement("div");
// 		botMsg.className = "message bot";
// 		botMsg.innerHTML = `<div class="bubble">Terima kasih! Pesan Anda telah diterima oleh ${
// 			method === "naivebayes" ? "Naive Bayes" : "CNN"
// 		}.</div>`;
// 		messages.appendChild(botMsg);

// 		input.value = "";
// 		selectedImages[method] = null;
// 		fileInput.value = "";
// 		messages.scrollTop = messages.scrollHeight;
// 	});
// }

// setupChat("naivebayes");
// setupChat("cnn");

// window.addEventListener("DOMContentLoaded", () => {
// 	const urlParams = new URLSearchParams(window.location.search);
// 	const metode = urlParams.get("metode");

// 	const chatNaiveBayes = document.getElementById("chat-naivebayes");
// 	const chatCNN = document.getElementById("chat-cnn");

// 	// Sembunyikan semua dulu
// 	chatNaiveBayes.classList.remove("active");
// 	chatCNN.classList.remove("active");

// 	// Tampilkan sesuai metode
// 	if (metode === "naivebayes") {
// 		chatNaiveBayes.classList.add("active");
// 	} else if (metode === "cnn") {
// 		chatCNN.classList.add("active");
// 	}
// });
