#!/usr/bin/env python3
"""Generate updated resume PDF for Ashish S."""

from fpdf import FPDF


class ResumePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header_block(self, name, contact_line, links):
        # Name
        self.set_font("Helvetica", "B", 22)
        self.cell(0, 10, name, new_x="LMARGIN", new_y="NEXT", align="C")
        # Contact
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, contact_line, new_x="LMARGIN", new_y="NEXT", align="C")
        # Links
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, links, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        # thin line
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def entry_header(self, left, right):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 5, left, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, right, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 9.5)
        x = self.get_x()
        self.cell(5, 5, "-", new_x="END", new_y="TOP")
        self.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)


pdf = ResumePDF()
pdf.add_page()
pdf.set_margins(18, 15, 18)

# --- Header ---
pdf.header_block(
    "Ashish S",
    "Chennai, India 600081  |  +91 81 4844 9229  |  ash2007in@gmail.com",
    "LinkedIn: linkedin.com/in/ashish-s-2857b2389  |  GitHub: github.com/Dinoman67",
)

# --- Professional Summary ---
pdf.section_title("Professional Summary")
pdf.set_font("Helvetica", "", 9.5)
pdf.multi_cell(0, 5,
    "B.Tech Machine Learning student with hands-on experience building and deploying "
    "deep learning models for real-world problems. Comfortable working across the full "
    "stack of an ML project, from data preparation and model training to deployment on "
    "edge devices. Looking for an internship to apply and grow these skills in a "
    "professional setting."
)
pdf.ln(3)

# --- Skills ---
pdf.section_title("Skills")
pdf.set_font("Helvetica", "", 9.5)
skills = [
    ("Languages", "Python, C"),
    ("ML / DL", "PyTorch, YOLOv8, ONNX Runtime, model quantization (FP16/INT8)"),
    ("NLP / Speech", "Whisper ASR, Sarvam STT, LLM fine-tuning (GGUF/llama.cpp)"),
    ("Web / APIs", "FastAPI, React, Vite, Google Colab"),
    ("Tools", "Git, Linux, Docker, edge deployment (Raspberry Pi)"),
]
for label, detail in skills:
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(30, 5, label + ":", new_x="END", new_y="TOP")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.multi_cell(0, 5, detail, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)
pdf.ln(2)

# --- Projects ---
pdf.section_title("Projects")

# Project 1: SonarVision
pdf.entry_header(
    "SonarVision - Underwater Marine Debris Detection",
    "GitHub: github.com/Dinoman67/sonarvision",
)
pdf.bullet(
    "Built YOLOv8-ESI, a lightweight object detection model with squeeze-and-excitation "
    "attention, optimized for side-scan sonar imagery."
)
pdf.bullet(
    "Achieved 88.4% mAP50 on unseen test data, a 12.3% improvement over baseline "
    "YOLOv8n, with only 3.3M parameters and a 6.2 MB model size."
)
pdf.bullet(
    "Exported the model to ONNX FP16 and deployed it on Raspberry Pi for real-time "
    "edge inference as part of the Smart India Hackathon 2026."
)
pdf.ln(2)

# Project 2: Rural Triage
pdf.entry_header(
    "Rural Triage - AI-Assisted Clinical Decision Support",
    "GitHub: github.com/Dinoman67/Rural-triage",
)
pdf.bullet(
    "Developed a voice-first multilingual triage platform for rural healthcare workers, "
    "using Tamil ASR, clinical NLP extraction, and rule-based urgency classification."
)
pdf.bullet(
    "Implemented a 6-stage pipeline: acoustic preprocessing, speech-to-text, translation, "
    "medical NLP, triage scoring, and SBAR summary generation."
)
pdf.bullet(
    "Built with FastAPI backend and React frontend; integrated Sarvam Saaras STT and "
    "Whisper for speech recognition, with fine-tuning on Mozilla Common Voice Tamil."
)
pdf.ln(2)

# Project 3: Legal Assistant LLM
pdf.entry_header(
    "Legal Assistant LLM - Fine-tuned Legal Language Model",
    "GitHub: github.com/Dinoman67/Legal-Assistant-LLM-GGUF",
)
pdf.bullet(
    "Fine-tuned a language model on legal Q&A data using instruction-based prompts for "
    "answering legal questions."
)
pdf.bullet(
    "Exported the model to GGUF format for efficient local inference using llama.cpp and "
    "Ollama, enabling offline legal query assistance."
)
pdf.ln(3)

# --- Experience ---
pdf.section_title("Experience")
pdf.entry_header(
    "Intern",
    "07/2025 - 08/2025  |  Zybeak Technologies - Chennai, India",
)
pdf.bullet("Analyzed problems and worked with teams to develop solutions.")
pdf.bullet(
    "Gained hands-on experience in various software programs, increasing proficiency "
    "and expanding technical skill set."
)
pdf.bullet(
    "Participated in workshops and presentations related to projects to gain knowledge."
)
pdf.bullet(
    "Developed organizational skills through managing multiple tasks simultaneously "
    "while adhering to strict deadlines."
)
pdf.ln(3)

# --- Education ---
pdf.section_title("Education")
pdf.entry_header(
    "Bachelor of Technology: Machine Learning",
    "Expected 08/2028  |  Saveetha Engineering College - Chennai, India",
)
pdf.ln(1)
pdf.entry_header(
    "High School Diploma",
    "05/2024  |  Thiruthangal Nadar Vidhyalaya - Chennai, India",
)

output_path = "/home/ashish/Downloads/Resume_Updated.pdf"
pdf.output(output_path)
print(f"Resume saved to {output_path}")
