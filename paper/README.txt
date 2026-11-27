Paper sources and reproduction
==============================

From the repository root, run python run.py figures to regenerate figures,
LaTeX tables and the CSV export from paper_data.json. Run python run.py report
to rebuild the PDF with Tectonic, latexmk or pdflatex. Both commands print
their output directory under runs/.

Each experiment owns its plot_*.py scripts under workspace/. The MountainCar
figure combines the source audit panel with the controlled-study learning curve.
make_figures.py assembles these outputs and writes the paper tables.

snapshot_data.py rebuilds the plotting snapshot from the full offline evidence.
audit_report.py and package_report.py are archival workflows requiring the
original run records and supporting files. qa_pdf.py checks a compiled
build/main.pdf and uses pdfplumber.
