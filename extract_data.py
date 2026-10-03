import json
import fitz

def dump_nb(path, out_file):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    with open(out_file, 'w', encoding='utf-8') as out:
        out.write(f"NOTEBOOK: {path}\n")
        out.write(f"Total cells: {len(nb.get('cells', []))}\n\n")
        for idx, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type', 'unknown')
            source = ''.join(cell.get('source', []))
            out.write(f"=== CELL {idx+1} [{cell_type.upper()}] ===\n")
            out.write(source + "\n")
            if cell_type == 'code':
                outputs = cell.get('outputs', [])
                if outputs:
                    out.write("--- OUTPUTS ---\n")
                    for o in outputs:
                        ot = o.get('output_type', '')
                        if ot == 'stream':
                            out.write(''.join(o.get('text', [])))
                        elif ot in ['execute_result', 'display_data']:
                            data = o.get('data', {})
                            if 'text/plain' in data:
                                out.write(''.join(data['text/plain']) + '\n')
                        elif ot == 'error':
                            out.write(f"ERROR: {o.get('ename')}: {o.get('evalue')}\n")
            out.write("\n" + "="*40 + "\n\n")

dump_nb('aasist-model.ipynb', 'scratch_aasist.txt')
dump_nb('ml-conformers-pa.ipynb', 'scratch_conformer.txt')
print("Notebooks dumped.")
