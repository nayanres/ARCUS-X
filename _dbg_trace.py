import post_run_analyzer as p

def _entry(z,g,t,mp,tp,tok=10):
    return (f'\n=== RAW STREAM ENTRY (Tokens: {tok}) ===\nParameters: z={z}, gravity={g}, tier={t}, grid=default\n------------------------------\n[MODEL OUTPUT]:\n{mp}\n\n[GROUND TRUTH EXPECTED]:\n{tp}\n\n==============================\n')

def _rb(rid,model,total,status,completed,failures,entries,reason=''):
    b='\n'+'='*50+'\nARCUS-X RUN START\nRun ID: '+rid+'\nModel: '+model+'\nTotal Probes: '+str(total)+'\n'+'='*50+'\n'
    b+=''.join(entries)
    b+='\n'+'='*50+'\nARCUS-X RUN END\nStatus: '+status+'\nCompleted: '+str(completed)+'/'+str(total)+'\nFailures: '+str(failures)+'\nEnd Time: t\n'+'='*50+'\n'
    return b

log=_rb('R1','m',2,'COMPLETE',2,0,[_entry(3,0.0,0,'[0,0]->[1,0]->[2,0]',"['[0,0]', '[1,0]', '[2,0]']")])

# Redirect stderr to file
import sys
old_stderr = sys.stderr
sys.stderr = open('outputs/_dbg_trace.txt', 'w', encoding='utf-8')
segs = p._split_runs(log)
sys.stderr.close()
sys.stderr = old_stderr

with open('outputs/_dbg_trace.txt', 'a', encoding='utf-8') as f:
    f.write(f"\nSEGS: {len(segs)}\n")
