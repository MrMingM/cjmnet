"""Read-only validation of a RUN; full integrity checks by default."""
import argparse
from .common import read_json,run_path,Manifest,digest,object_hash,verify_snapshot,verify_protocol

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True)
    p.add_argument('--quick',action='store_true',help='Progress only; does not assert full integrity')
    args=p.parse_args(); run=run_path(args.run)
    if not (run/'protocol.json').exists():
        print('Protocol not published yet: preflight/hashing may still be running or failed. Inspect driver.log and diagnostics/failure.json.')
        raise SystemExit(2)
    protocol=read_json(run/'protocol.json'); verify_protocol(protocol)
    m=Manifest(run,protocol['identity'])
    expected=len(protocol['selection'])*len(protocol['config']['weathers'])
    completed=sum(k.startswith('source_observability/') for k in m.value['artifacts'])
    print(f"stage={m.value['stage']} extracted={completed}/{expected} mode={protocol['mode']}")
    if args.quick:
        print('Quick progress only; integrity not verified.'); return
    if object_hash(verify_snapshot())!=protocol['source_hash']:
        raise ValueError('Source identity mismatch')
    for key in m.value['artifacts']:
        m.get(key)
    if m.value.get('environment_hash') and digest(run/'environment.json')!=m.value['environment_hash']:
        raise ValueError('Environment record changed')
    from .data import verify_files
    verify_files(protocol['dataset'])
    for stem in ('config','checkpoint'):
        if digest(protocol['frontend'][stem+'_path'])!=protocol['frontend'][stem+'_sha256']:
            raise ValueError('Frontend artifact changed')
    if m.value['stage']=='complete':
        from .pipeline import artifact_key
        for weather in protocol['config']['weathers']:
            for record in protocol['selection']:
                for folder in ('source_observability','subset_predictions','diagnostics/frames'):
                    if m.get(folder+'/'+artifact_key(record,weather)) is None:
                        raise ValueError('Missing expected artifact')
        hashes=m.value.get('final_hashes',{})
        if len(hashes)!=6:
            raise ValueError('Missing final reports')
        for key,value in hashes.items():
            if digest(run/key)!=value:
                raise ValueError('Report modified: '+key)
        print('All hashes and expected artifacts verified.',m.value['decision'])
    else:
        print('Existing artifacts verified; RUN is not complete.')
        raise SystemExit(2)

if __name__=='__main__':
    main()
