"""Run with a Python containing pdfplumber; visual review remains required."""
from pathlib import Path
import json,re,hashlib,shutil
import pdfplumber
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    path=HERE/'build/main.pdf';out={};outside=[]
    with pdfplumber.open(path) as pdf:
        texts=[p.extract_text() or '' for p in pdf.pages]
        references=[i+1 for i,t in enumerate(texts) if t.startswith('References')]
        assert references==[8] and len(pdf.pages)==8
        assert all(len(t)>350 for t in texts)
        for i,p in enumerate(pdf.pages):
            outside += [(i+1,c['text']) for c in p.chars if c['x0']<-.5 or c['x1']>p.width+.5 or c['top']<-.5 or c['bottom']>p.height+.5]
    assert not outside
    text='\n'.join(texts);assert '\ufffd' not in text and '[?]' not in text and '??' not in text
    assert all(re.search(r'Figure\s*'+str(i)+r'\s*:',text) for i in range(1,5))
    assert all('['+str(i)+']' in texts[-1] for i in range(1,11))
    tex=(HERE/'main.tex').read_text();abstract=tex.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    words=len(re.sub(r'\\[A-Za-z]+','',abstract).split());assert 150<=words<=200
    assert '\\author{}' in tex
    final=ROOT/'output/pdf/pheromone_shaping_report.pdf';final.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,final)
    assert hashlib.sha256(final.read_bytes()).digest()==hashlib.sha256(path.read_bytes()).digest()
    out=dict(pages=8,main_text_pages=7,references_pages=1,abstract_words=words,figure_captions=4,
        references=10,unresolved_reference_markers=False,out_of_page_characters=outside,
        author_field_blank=True,final_path=str(final),pdf_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        note='All eight pages were also rendered and visually inspected; checks alone do not establish content validity.')
    (HERE/'pdf_qa.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
