"""
test_parser.py - unit tests for the LaTeX resume parser.

Tests that the parser extracts all sections: header, skills, experience,
education, projects, achievements, certifications, publications, and extras.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.parser.tex_parser import parse_tex_string
from app.core.ir import ResumeIR


# -- Minimal .tex samples for testing --

MINIMAL_TEX = r"""
\documentclass{article}
\begin{document}

\begin{center}
{\Huge \textbf{Jane Smith}} \\
\small \href{mailto:jane@example.com}{jane@example.com}
\quad \href{https://linkedin.com/in/janesmith}{linkedin.com/in/janesmith}
\quad \href{https://github.com/janesmith}{github.com/janesmith}
\end{center}

\section{Professional Summary}
Senior ML engineer with 5 years building NLP and CV systems on Azure.

\section{Technical Skills}
\begin{itemize}[leftmargin=0.15in, label={}]
    \small{\item{
     \textbf{Languages}{: Python, C++, SQL} \\
     \textbf{ML Frameworks}{: PyTorch, TensorFlow, scikit-learn} \\
     \textbf{Cloud}{: Azure, AWS, Docker}
    }}
\end{itemize}

\section{Experience}
  \resumeSubHeadingListStart
    \resumeSubheading
      {ML Engineer}{Jan 2022 -- Present}
      {Acme Corp}{San Francisco, CA}
      \resumeItemListStart
        \resumeItem{Built a recommendation engine using FAISS and Azure AI Search}
        \resumeItem{Deployed models via Triton Inference Server with TensorRT optimization}
      \resumeItemListEnd
  \resumeSubHeadingListEnd

\section{Education}
\begin{itemize}[leftmargin=0.15in, label={}]
  \item
    \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{M.S. Computer Science} & \textit{2020 -- 2022} \\
      {\small MIT} & {\small Cambridge, MA} \\
    \end{tabular*}
\end{itemize}

\section{Projects}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Resume Enhancer}{2024}
      {Multi-agent resume optimization tool}{Gradio, Jinja2}
      \resumeItemListStart
        \resumeItem{Built a Gradio UI for real-time agent trace visualization}
      \resumeItemListEnd
  \resumeSubHeadingListEnd

\section{Achievements}
\achieveRow{Best Paper}{Top paper at internal ML summit}{2023}
\achieveRow{Hackathon Winner}{Built AI chatbot in 48 hours}{2022}

\section{Certifications}
\achieveRow{AWS Solutions Architect}{Amazon Web Services}{2023}

\end{document}
"""

MINIMAL_TEX_WITH_PUBLICATIONS = MINIMAL_TEX.replace(
    r"\end{document}",
    r"""
\section{Publications}
\begin{itemize}
\item On Neural Architecture Search. \textit{NeurIPS 2023}.
\end{itemize}

\end{document}
"""
)


class TestParserBasic:
    def test_parses_without_error(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert isinstance(ir, ResumeIR)

    def test_extracts_name(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert ir.header.name
        assert "Jane" in ir.header.name or "Smith" in ir.header.name

    def test_extracts_links(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert len(ir.header.links) > 0
        kinds = [link.kind for link in ir.header.links]
        assert "email" in kinds or any("jane" in l.label for l in ir.header.links)

    def test_extracts_summary(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert ir.summary
        assert "ML" in ir.summary or "engineer" in ir.summary.lower()

    def test_extracts_skills(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert len(ir.skills) > 0
        all_items = []
        for b in ir.skills:
            all_items.extend(b.items)
        assert any("Python" in item for item in all_items)

    def test_extracts_experience(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert len(ir.experience) > 0
        exp = ir.experience[0]
        assert exp.bullets
        assert len(exp.bullets) >= 1

    def test_extracts_education(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert len(ir.education) > 0

    def test_extracts_projects(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert len(ir.projects) > 0

    def test_extracts_achievements(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert len(ir.achievements) > 0

    def test_extracts_certifications(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert len(ir.certifications) > 0

    def test_section_order_has_defaults(self):
        ir = parse_tex_string(MINIMAL_TEX)
        assert isinstance(ir.section_order, list)
        assert "summary" in ir.section_order
        assert "experience" in ir.section_order

    def test_text_blob_contains_all(self):
        ir = parse_tex_string(MINIMAL_TEX)
        blob = ir.text_blob()
        assert "Python" in blob
        assert "FAISS" in blob or "recommendation" in blob.lower()


class TestParserCompleteness:
    """Ensure nothing is left behind during extraction."""

    def test_empty_input_doesnt_crash(self):
        ir = parse_tex_string("")
        assert isinstance(ir, ResumeIR)

    def test_no_sections_doesnt_crash(self):
        ir = parse_tex_string(r"\documentclass{article}\begin{document}Hello\end{document}")
        assert isinstance(ir, ResumeIR)

    def test_publications_extracted(self):
        ir = parse_tex_string(MINIMAL_TEX_WITH_PUBLICATIONS)
        # Publications may be extracted into publications list or extras
        has_pubs = (
            len(ir.publications) > 0
            or any("publication" in x.title.lower() for x in ir.extras)
        )
        assert has_pubs


class TestPipelineConfig:
    def test_placeholder_detection(self):
        from app.pipeline import _looks_like_placeholder
        assert _looks_like_placeholder("[PLACEHOLDER]")
        assert _looks_like_placeholder("  [YOUR NAME HERE]  ")
        assert not _looks_like_placeholder("[Company Name] is a leading provider")
        assert not _looks_like_placeholder("This is normal text")
        assert not _looks_like_placeholder("[a lowercase placeholder]")
