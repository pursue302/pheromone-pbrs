"""Execute results-only Python cells in IPython; never starts training.

This notebook has no magics or kernel-specific operations. In-process execution
avoids Windows kernel transport startup and still evaluates every code cell.
"""
from pathlib import Path
import os

HERE=Path(__file__).resolve().parent
os.environ['IPYTHONDIR']=str(HERE/'.ipython_results')
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

path=HERE/'Pheromone_feedback_2019.ipynb'
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
for i,cell in enumerate(nb.cells):
    if cell.cell_type=='code':
        compile(cell.source,f'notebook-cell-{i}','exec')
shell=InteractiveShell.instance()
count=0
for cell in nb.cells:
    if cell.cell_type!='code':
        continue
    count+=1
    with capture_output() as captured:
        result=shell.run_cell(cell.source,store_history=False)
    error=result.error_before_exec or result.error_in_exec
    if error:
        raise RuntimeError(f'Notebook cell {count}: {error}')
    outputs=[]
    if captured.stdout:
        outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:
        outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for rich in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    cell.execution_count=count
    cell.outputs=outputs
nb.metadata['execution_method']='IPython in-process, all Python cells evaluated'
nbformat.validate(nb)
nbformat.write(nb,path)
errors=[output for cell in nb.cells if cell.cell_type=='code' for output in cell.outputs
        if output.output_type=='error']
assert not errors, errors
print('Executed and validated',len(nb.cells),'cells; notebook bytes:',path.stat().st_size)
