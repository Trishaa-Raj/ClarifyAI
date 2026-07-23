/* ============================================================
   ClarifyAI — Frontend Script
   Handles: PDF upload, question sending, response rendering

   FIXES APPLIED:
   1. API_BASE hardcoded localhost → relative URL (works locally + deployed)
   2. XSS: all AI-generated text fields escaped before innerHTML injection
============================================================ */

// FIX 1: Relative URL — no hardcoded host or port.
// Works on localhost, works on Render, works on any deployment.
// Before: const API_BASE = "http://localhost:8000";  ← breaks on deploy
const API_BASE = "";

// DOM references
const chatBox       = document.getElementById("chatBox");
const questionInput = document.getElementById("question");
const sendBtn       = document.getElementById("sendBtn");
const processBtn    = document.getElementById("processBtn");
const fileInput     = document.getElementById("pdfUpload");
const fileNameEl    = document.getElementById("fileName");
const docStatusDot  = document.querySelector(".status-dot");
const docStatusText = document.getElementById("docStatusText");

// ============================================================
// FIX 2: HTML ESCAPE UTILITY
// Run every string that came from an API or user through this
// before placing it inside innerHTML.
//
// What it does:
//   <img src=x onerror=alert(1)>
//   → &lt;img src=x onerror=alert(1)&gt;
//
// The browser renders it as visible text — never executes it.
//
// Rule: innerHTML is safe ONLY for strings you wrote yourself
// in source code. Anything from an API, file, or user = escape first.
// ============================================================
function escapeHTML(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g,  "&amp;")   // must be first — prevents double-escaping
        .replace(/</g,  "&lt;")    // kills all HTML tags
        .replace(/>/g,  "&gt;")
        .replace(/"/g,  "&quot;")
        .replace(/'/g,  "&#x27;");
}

// ============================================================
// FILE SELECTION
// When user picks a file, show its name and enable Process button.
// ============================================================
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        fileNameEl.textContent = file.name;
        fileNameEl.classList.add("selected");
        processBtn.disabled = false;
    } else {
        fileNameEl.textContent = "No file selected";
        fileNameEl.classList.remove("selected");
        processBtn.disabled = true;
    }
});

// ============================================================
// PDF UPLOAD
// Sends the selected PDF to the backend /upload endpoint.
// Uses relative URL now — API_BASE is "".
// ============================================================
async function uploadPDF() {
    const file = fileInput.files[0];

    if (!file) {
        addStatusMessage("Please select a PDF file first.", "error");
        return;
    }

    setDocStatus("loading", `Processing "${file.name}"...`);
    processBtn.disabled = true;
    processBtn.textContent = "Processing...";
    clearWelcome();

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed");
        }

        setDocStatus("active", `"${file.name}" ready`);
        addStatusMessage(`✅ ${data.message}`, "success");

        questionInput.disabled = false;
        sendBtn.disabled = false;
        questionInput.focus();

    } catch (error) {
        setDocStatus("error", "Processing failed");
        addStatusMessage(`❌ ${error.message}`, "error");
        processBtn.disabled = false;
    } finally {
        // FIX: always reset button text — even if an error is thrown
        processBtn.textContent = "Process";
    }
}

// ============================================================
// SEND QUESTION
// Sends the user's question + level to /ask endpoint.
// ============================================================
async function sendMessage() {
    const question = questionInput.value.trim();
    const level = document.getElementById("level").value;

    if (!question) return;

    addUserMessage(question);
    questionInput.value = "";
    setInputLocked(true);

    const typingEl = addTypingIndicator();

    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, level })
        });

        const data = await response.json();
        removeTypingIndicator(typingEl);

        if (!response.ok) {
            throw new Error(data.detail || "Server error");
        }

        const answer = data.answer || {};

        if (answer.error) {
            // FIX: escape API error strings before display
            addErrorMessage(`Model error: ${escapeHTML(answer.error)}. ${escapeHTML(answer.details || "")}`);
        } else if (answer.raw_response) {
            // FIX: raw_response is unstructured LLM text — must be escaped
            addBotMessage(`<p>${escapeHTML(answer.raw_response)}</p>`);
        } else {
            addStructuredAnswer(answer);
        }

    } catch (error) {
        removeTypingIndicator(typingEl);
        addErrorMessage(`Request failed: ${error.message}`);
    }

    setInputLocked(false);
    questionInput.focus();
}

// ============================================================
// RENDER STRUCTURED ANSWER
// FIX: Every AI-returned text field is now escaped before being
// injected into innerHTML.
//
// Fields that come from the LLM (unsafe — must escape):
//   main_idea, real_world_example, simple_summary,
//   equations_explained, key_concepts[].concept,
//   key_concepts[].explanation
//
// Fields we write ourselves (safe — no escaping needed):
//   section labels like "📌 Main Idea", HTML structure tags
// ============================================================
function addStructuredAnswer(answer) {
    let html = "";

    if (answer.main_idea) {
        // FIX: escapeHTML wraps LLM text — tags become visible text, not code
        html += answerSection("📌 Main Idea", `<p>${escapeHTML(answer.main_idea)}</p>`);
    }

    if (answer.key_concepts && Array.isArray(answer.key_concepts) && answer.key_concepts.length > 0) {
        const items = answer.key_concepts.map(c => {
            if (typeof c === "string") {
                // FIX: plain string concept — escape it
                return `<li>${escapeHTML(c)}</li>`;
            }
            const term = c.concept || c.term || "";
            const exp  = c.explanation || "";
            // FIX: both term and explanation come from LLM — escape both
            return `<li>${term ? `<strong>${escapeHTML(term)}:</strong> ` : ""}${escapeHTML(exp)}</li>`;
        }).join("");
        html += answerSection("💡 Key Concepts", `<ul>${items}</ul>`);
    }

    if (answer.equations_explained && answer.equations_explained !== "N/A") {
        // FIX: escape equation text — LLM sometimes wraps in HTML tags
        html += answerSection("🧮 Equations", `<p>${escapeHTML(answer.equations_explained)}</p>`);
    }

    if (answer.real_world_example && answer.real_world_example !== "N/A") {
        // FIX: escape example text
        html += answerSection("🌍 Real-World Example", `<p>${escapeHTML(answer.real_world_example)}</p>`);
    }

    if (answer.simple_summary) {
        // FIX: escape summary text
        html += answerSection("📝 Summary", `<p>${escapeHTML(answer.simple_summary)}</p>`);
    }

    if (!html) {
        // Fallback for unexpected JSON shape — escape the raw dump too
        html = `<p style="color:#6b7280">The AI returned an unexpected response format.</p>
                <pre style="font-size:11px;overflow:auto">${escapeHTML(JSON.stringify(answer, null, 2))}</pre>`;
    }

    addBotMessage(html);
}

// Helper: wraps content in a labelled answer section
// Label strings ("📌 Main Idea") are written by us — safe, no escaping needed
// contentHTML is already built with escaped values above
function answerSection(label, contentHTML) {
    return `
        <div class="answer-section">
            <div class="answer-label">${label}</div>
            ${contentHTML}
        </div>
    `;
}

// ============================================================
// MESSAGE HELPERS
// ============================================================

function clearWelcome() {
    const welcome = chatBox.querySelector(".welcome-message");
    if (welcome) welcome.remove();
}

function addUserMessage(text) {
    clearWelcome();
    const div = document.createElement("div");
    div.className = "message user";
    div.textContent = text; // textContent — never innerHTML for user input
    chatBox.appendChild(div);
    scrollToBottom();
}

function addBotMessage(html) {
    clearWelcome();
    const div = document.createElement("div");
    div.className = "message bot";
    div.innerHTML = html; // safe: all dynamic values escaped before reaching here
    chatBox.appendChild(div);
    scrollToBottom();
}

function addStatusMessage(text, type = "normal") {
    clearWelcome();
    const div = document.createElement("div");
    div.className = "message bot status-msg";
    if (type === "error") div.classList.add("error-msg");
    div.textContent = text;
    chatBox.appendChild(div);
    scrollToBottom();
}

function addErrorMessage(text) {
    clearWelcome();
    const div = document.createElement("div");
    div.className = "message bot error-msg";
    div.textContent = `❌ ${text}`;
    chatBox.appendChild(div);
    scrollToBottom();
}

function addTypingIndicator() {
    clearWelcome();
    const div = document.createElement("div");
    div.className = "typing-indicator";
    // Safe: hardcoded string written by us, no external data
    div.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    chatBox.appendChild(div);
    scrollToBottom();
    return div;
}

function removeTypingIndicator(el) {
    if (el && el.parentNode) el.remove();
}

// ============================================================
// UI STATE HELPERS
// ============================================================

function setDocStatus(state, text) {
    docStatusDot.className = `status-dot ${state}`;
    docStatusText.textContent = text;
}

function setInputLocked(locked) {
    questionInput.disabled = locked;
    sendBtn.disabled = locked;
}

function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
}

// ============================================================
// KEYBOARD: Enter to send
// ============================================================
questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) sendMessage();
    }
});
