"""
Kaggle Dataset Downloader & Path Resolver for ASVspoof 2019.
Automatically downloads via kagglehub and resolves the directory structure.

Usage:
    from audio_deepfake_detector.kaggle_data import download_asvspoof2019, get_la_paths
    
    data_root = download_asvspoof2019()
    paths = get_la_paths(data_root)
    # paths['train_protocol'], paths['train_audio'], etc.
"""

import os
import glob


def download_asvspoof2019():
    """
    Downloads ASVspoof 2019 dataset via kagglehub.
    
    Returns:
        str: Path to the root of the downloaded dataset.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        )
    
    print("[KaggleHub] Downloading ASVspoof 2019 dataset...")
    path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
    print(f"[KaggleHub] Dataset path: {path}")
    return path


def find_file_recursive(root, pattern):
    """
    Recursively search for a file/dir matching a pattern under root.
    Returns first match or None.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            if pattern in name:
                return os.path.join(dirpath, name)
    return None


def get_la_paths(data_root):
    """
    Resolves all ASVspoof 2019 LA protocol and audio directory paths
    from the kagglehub download root.
    
    The Kaggle dataset "awsaf49/asvpoof-2019-dataset" has structure:
        <root>/LA/LA/ASVspoof2019_LA_train/flac/
        <root>/LA/LA/ASVspoof2019_LA_dev/flac/
        <root>/LA/LA/ASVspoof2019_LA_eval/flac/  (may not exist in all versions)
        <root>/LA/LA/ASVspoof2019_LA_cm_protocols/
            ASVspoof2019.LA.cm.train.trn.txt
            ASVspoof2019.LA.cm.dev.trl.txt
            ASVspoof2019.LA.cm.eval.trl.txt
    
    Args:
        data_root (str): Root path from kagglehub.dataset_download()
    
    Returns:
        dict: {
            'la_root':        str,  # Base LA directory
            'train_protocol': str,  # Protocol txt file for training
            'dev_protocol':   str,  # Protocol txt file for dev
            'eval_protocol':  str,  # Protocol txt file for evaluation
            'train_audio':    str,  # Audio flac directory for training
            'dev_audio':      str,  # Audio flac directory for dev
            'eval_audio':     str,  # Audio flac directory for evaluation
            'has_eval':       bool, # Whether eval partition exists
        }
    """
    # Try common directory structures
    la_root = None
    
    # Pattern 1: <root>/LA/LA/
    candidate = os.path.join(data_root, "LA", "LA")
    if os.path.isdir(candidate):
        la_root = candidate
    
    # Pattern 2: <root>/LA/
    if la_root is None:
        candidate = os.path.join(data_root, "LA")
        if os.path.isdir(candidate):
            la_root = candidate
    
    # Pattern 3: Search recursively for ASVspoof2019_LA_cm_protocols
    if la_root is None:
        proto_dir = find_file_recursive(data_root, "ASVspoof2019_LA_cm_protocols")
        if proto_dir:
            la_root = os.path.dirname(proto_dir)
    
    # Pattern 4: data_root itself contains the files
    if la_root is None:
        if os.path.exists(os.path.join(data_root, "ASVspoof2019_LA_cm_protocols")):
            la_root = data_root
    
    if la_root is None:
        # List what we found for debugging
        print(f"[WARNING] Could not find LA directory structure in: {data_root}")
        print(f"[WARNING] Contents of data_root:")
        if os.path.isdir(data_root):
            for item in os.listdir(data_root):
                print(f"   {item}")
        raise FileNotFoundError(
            f"Cannot find ASVspoof2019 LA directory structure under: {data_root}\n"
            f"Expected structure: <root>/LA/LA/ASVspoof2019_LA_cm_protocols/"
        )
    
    print(f"[Dataset] Found LA root: {la_root}")
    
    # Protocol files
    proto_dir = os.path.join(la_root, "ASVspoof2019_LA_cm_protocols")
    train_protocol = os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    dev_protocol   = os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
    eval_protocol  = os.path.join(proto_dir, "ASVspoof2019.LA.cm.eval.trl.txt")
    
    # Audio directories
    train_audio = os.path.join(la_root, "ASVspoof2019_LA_train", "flac")
    dev_audio   = os.path.join(la_root, "ASVspoof2019_LA_dev", "flac")
    eval_audio  = os.path.join(la_root, "ASVspoof2019_LA_eval", "flac")
    
    # Check what exists
    has_eval = os.path.isdir(eval_audio) and os.path.isfile(eval_protocol)
    
    # Report status
    checks = [
        ("Train Protocol", train_protocol),
        ("Dev Protocol",   dev_protocol),
        ("Eval Protocol",  eval_protocol),
        ("Train Audio",    train_audio),
        ("Dev Audio",      dev_audio),
        ("Eval Audio",     eval_audio),
    ]
    
    for name, path in checks:
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status}  {name}: {path}")
    
    return {
        'la_root':        la_root,
        'train_protocol': train_protocol,
        'dev_protocol':   dev_protocol,
        'eval_protocol':  eval_protocol,
        'train_audio':    train_audio,
        'dev_audio':      dev_audio,
        'eval_audio':     eval_audio,
        'has_eval':       has_eval,
    }


def quick_dataset_summary(paths):
    """
    Prints a quick summary of the dataset: file counts, partition sizes.
    """
    for partition in ['train', 'dev', 'eval']:
        proto = paths.get(f'{partition}_protocol', '')
        audio = paths.get(f'{partition}_audio', '')
        
        n_proto = 0
        n_bona  = 0
        n_spoof = 0
        
        if os.path.isfile(proto):
            with open(proto, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        n_proto += 1
                        if parts[4].lower() == 'bonafide':
                            n_bona += 1
                        else:
                            n_spoof += 1
        
        n_audio = 0
        if os.path.isdir(audio):
            n_audio = len([f for f in os.listdir(audio) if f.endswith('.flac')])
        
        print(f"\n  [{partition.upper()}]")
        print(f"    Protocol entries : {n_proto:>8,}")
        print(f"    Bonafide         : {n_bona:>8,}")
        print(f"    Spoof            : {n_spoof:>8,}")
        print(f"    Audio files      : {n_audio:>8,}")


if __name__ == "__main__":
    print("=" * 60)
    print("  ASVspoof 2019 LA Dataset - Kaggle Download & Verify")
    print("=" * 60)
    
    data_root = download_asvspoof2019()
    paths = get_la_paths(data_root)
    
    print("\n  Dataset Summary:")
    quick_dataset_summary(paths)
    
    print("\n" + "=" * 60)
    print("  Dataset ready for training pipeline!")
    print("=" * 60)
