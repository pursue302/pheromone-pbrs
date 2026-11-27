Independent navigation validation
=================================

experiments/engine.py runs the validation protocol. From the repository root:
  python run.py train validation --out validation-full
  python run.py analyze fresh --input runs/validation-full --out validation-stats

plot_navigation.py draws the navigation maps and fresh validation figure.
Use python run.py figures to regenerate the published figures and tables.
Paper sources and publication tooling live under paper/.
