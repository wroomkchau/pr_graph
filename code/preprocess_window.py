# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os, re, glob
import numpy as np
import torch
import time

MAX_EVENTS_PER_FILE   = int(os.environ.get('MAX_EVENTS_PER_FILE', '600000'))
MAX_WINDOWS_PER_FILE  = int(os.environ.get('MAX_WINDOWS_PER_FILE', '80'))
WIN_EVENTS            = int(os.environ.get('WIN_EVENTS', '8000'))
STRIDE_EVENTS         = int(os.environ.get('STRIDE_EVENTS', '4000'))
MAX_NODES             = int(os.environ.get('MAX_NODES', '1000'))
SLEEP_SEC             = float(os.environ.get('SLEEP_SEC', '0.005'))


def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def letter_from_filename(path):
    base = os.path.basename(path)
    m = re.match(r'^([a-zA-Z])', base)
    return m.group(1).lower() if m else None

def build_label_map(all_aedat_paths):
    letters = []
    for p in all_aedat_paths:
        l = letter_from_filename(p)
        if l is not None:
            letters.append(l)
    letters = sorted(list(set(letters)))
    return {l:i for i,l in enumerate(letters)}

def skip_aedat_header(fh):
    pos = 0
    while True:
        line = fh.readline()
        if not line:
            break
        if isinstance(line, bytes):
            is_hdr = line.startswith(b'#')
        else:
            is_hdr = line.startswith('#')
        if is_hdr:
            pos = fh.tell()
            continue
        fh.seek(pos, 0)
        break

def read_bytes(path):
    with open(path, 'rb') as fh:
        skip_aedat_header(fh)
        return fh.read()

def parse_events_addr_ts(raw):
    # require multiple of 8 bytes
    n = (len(raw) // 8) * 8
    if n < 8:
        return None, None
    raw = raw[:n]
    try:
        be = np.frombuffer(raw, dtype='>u4').reshape((-1, 2))
        le = np.frombuffer(raw, dtype='<u4').reshape((-1, 2))
    except Exception:
        return None, None
    return be, le

def score_ts(ts):
    if ts is None or len(ts) < 3:
        return -1
    d = np.diff(ts.astype(np.int64))
    return float(np.mean(d >= 0))


# Address decoders (candidates)

def decode_dvs128(addr):
    x = (addr >> 1) & 0x7F
    y = (addr >> 8) & 0x7F
    p = addr & 0x1
    return x, y, p, 128, 128

def decode_davis240(addr):
    x = (addr >> 17) & 0x1FF
    y = (addr >> 2)  & 0xFF
    p = (addr >> 1)  & 0x1
    return x, y, p, 240, 180

def decode_dvs346(addr):
    x = (addr >> 17) & 0x3FF
    y = (addr >> 2)  & 0x3FF
    p = (addr >> 1)  & 0x1
    return x, y, p, 346, 260

DECODERS = [
    ('dvs128', decode_dvs128),
    ('davis240', decode_davis240),
    ('dvs346', decode_dvs346),
]

def score_xy(x, y, W_hint, H_hint):
    if x is None or y is None or len(x) < 10:
        return -1.0
    x = x.astype(np.int64); y = y.astype(np.int64)
    ux = len(np.unique(x[:5000])); uy = len(np.unique(y[:5000]))
    if ux <= 1 or uy <= 1:
        return -1.0
    xmin, xmax = int(x.min()), int(x.max())
    ymin, ymax = int(y.min()), int(y.max())
    if xmax - xmin > 4096 or ymax - ymin > 4096:
        return -1.0
    bonus = 0.0
    if xmax <= W_hint + 50 and ymax <= H_hint + 50:
        bonus += 0.5
    return (ux/50.0 + uy/50.0) + bonus

def decode_best(addr, ts):
    best = None; best_name = None; best_score = -1e9
    for name, fn in DECODERS:
        x, y, p, W, H = fn(addr)
        s = score_xy(x, y, W, H)
        if s > best_score:
            best_score = s; best = (x, y, p); best_name = name
    if best is None or best_score < 0:
        return None, None
    return best_name, best

def read_aedat_events(path):
    try:
        raw = read_bytes(path)
    except Exception:
        return None

    be, le = parse_events_addr_ts(raw)
    if be is None:
        return None

    addr_be, ts_be = be[:,0], be[:,1]
    addr_le, ts_le = le[:,0], le[:,1]
    if score_ts(ts_le) > score_ts(ts_be):
        addr, ts = addr_le, ts_le
    else:
        addr, ts = addr_be, ts_be

    # limit events per file (VERY IMPORTANT for big files)
    if len(addr) > MAX_EVENTS_PER_FILE:
        addr = addr[:MAX_EVENTS_PER_FILE]
        ts   = ts[:MAX_EVENTS_PER_FILE]

    dec_name, decoded = decode_best(addr, ts)
    if decoded is None:
        return None

    x, y, p = decoded
    return {
        'decoder': dec_name,
        'x': x.astype(np.int64),
        'y': y.astype(np.int64),
        'p': p.astype(np.int64),
        't': ts.astype(np.int64),
    }

# Graph + windows

def build_graph_from_window(x, y, p, t):
    n = len(t)
    if n < 50:
        return None

    order = np.argsort(t)
    x = x[order]; y = y[order]; p = p[order]; t = t[order]

    if n > MAX_NODES:
        idx = np.linspace(0, n-1, MAX_NODES).astype(np.int64)
        x = x[idx]; y = y[idx]; p = p[idx]; t = t[idx]
        n = len(t)

    x_rng = float(np.max(x) - np.min(x))
    y_rng = float(np.max(y) - np.min(y))
    t_rng = float(np.max(t) - np.min(t))
    if x_rng <= 0 or y_rng <= 0 or t_rng <= 0:
        return None

    xf = (x - np.min(x)) / (x_rng + 1e-6)
    yf = (y - np.min(y)) / (y_rng + 1e-6)
    tf = (t - np.min(t)) / (t_rng + 1e-6)
    pf = p.astype(np.float32)

    feat = np.stack([xf, yf, tf, pf], axis=1).astype(np.float32)

    src = np.arange(0, n-1, dtype=np.int64)
    dst = np.arange(1, n, dtype=np.int64)
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)

    return {'x': torch.from_numpy(feat), 'edge_index': torch.from_numpy(edge_index).long()}

def make_windows(ev):
    x = ev['x']; y = ev['y']; p = ev['p']; t = ev['t']
    n = len(t)
    out = []
    start = 0
    while start + WIN_EVENTS <= n and len(out) < MAX_WINDOWS_PER_FILE:
        sl = slice(start, start + WIN_EVENTS)
        out.append((x[sl], y[sl], p[sl], t[sl]))
        start += STRIDE_EVENTS
    return out


# Resume-safe save

def safe_save_pt(obj, out_path):
    done_path = out_path + ".done"
    tmp_path  = out_path + ".tmp"

    if os.path.exists(done_path):
        return 'skip_done'

    if os.path.exists(out_path) and (not os.path.exists(done_path)):
        try: os.remove(out_path)
        except Exception: pass

    if os.path.exists(tmp_path):
        try: os.remove(tmp_path)
        except Exception: pass

    torch.save(obj, tmp_path)
    os.rename(tmp_path, out_path)
    with open(done_path, 'w') as f:
        f.write("ok\n")
    return 'saved'


# Main

def process_subject(subject_dir, out_root, label_map, split_name):
    processed_dir = os.path.join(out_root, 'processed')
    ensure_dir(processed_dir)

    aedat_files = sorted(glob.glob(os.path.join(subject_dir, '*.aedat')))
    saved = bad = skipped = resumed = 0

    for path in aedat_files:
        letter = letter_from_filename(path)
        if letter is None or letter not in label_map:
            skipped += 1
            continue
        y_label = int(label_map[letter])

        ev = read_aedat_events(path)
        if ev is None:
            print("[BAD] cannot decode:", path)
            bad += 1
            continue

        if len(np.unique(ev['x'][:5000])) <= 1 or len(np.unique(ev['y'][:5000])) <= 1:
            print("[SKIP] degenerate coords:", path, "(decoder:", ev['decoder'] + ")")
            skipped += 1
            continue

        windows = make_windows(ev)
        if len(windows) == 0:
            print("[SKIP] no windows:", path, "(events:", len(ev['t']), "decoder:", ev['decoder'] + ")")
            skipped += 1
            continue

        base = os.path.splitext(os.path.basename(path))[0]
        for wi, (x,y,p,t) in enumerate(windows):
            out_name = "%s_%s_%s_w%03d.pt" % (split_name, os.path.basename(subject_dir), base, wi)
            out_path = os.path.join(processed_dir, out_name)

            g = build_graph_from_window(x,y,p,t)
            if g is None:
                continue
            g['y'] = torch.tensor([y_label]).long()

            st = safe_save_pt(g, out_path)
            if st == 'saved':
                saved += 1
                time.sleep(SLEEP_SEC)
            else:
                resumed += 1

    return saved, bad, skipped, resumed

def maybe_clean_dir(p):
    if os.environ.get('CLEAN', '0') != '1':
        return
    if not os.path.isdir(p):
        return
    for fn in os.listdir(p):
        fp = os.path.join(p, fn)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
        except Exception:
            pass

def main():
    base_in = os.path.join('..', 'data', 'AEDAT')
    train_out = os.path.join('..', 'data', 'Traingraph')
    test_out  = os.path.join('..', 'data', 'Testgraph')

    subjects_train = ['Subject1', 'Subject2', 'Subject3', 'Subject4']
    subjects_test  = ['Subject5']

    all_files = glob.glob(os.path.join(base_in, 'Subject*', '*.aedat'))
    label_map = build_label_map(all_files)
    print("Label map:", label_map)
    print("Limits: MAX_EVENTS_PER_FILE=%d MAX_WINDOWS_PER_FILE=%d WIN_EVENTS=%d STRIDE_EVENTS=%d MAX_NODES=%d" %
          (MAX_EVENTS_PER_FILE, MAX_WINDOWS_PER_FILE, WIN_EVENTS, STRIDE_EVENTS, MAX_NODES))

    train_proc = os.path.join(train_out, 'processed')
    test_proc  = os.path.join(test_out, 'processed')
    ensure_dir(train_proc); ensure_dir(test_proc)

    maybe_clean_dir(train_proc)
    maybe_clean_dir(test_proc)

    train_saved = train_bad = train_skipped = train_resumed = 0
    test_saved  = test_bad  = test_skipped  = test_resumed  = 0

    for s in subjects_train:
        sd = os.path.join(base_in, s)
        a,b,c,r = process_subject(sd, train_out, label_map, split_name='TRAIN')
        train_saved += a; train_bad += b; train_skipped += c; train_resumed += r

    for s in subjects_test:
        sd = os.path.join(base_in, s)
        a,b,c,r = process_subject(sd, test_out, label_map, split_name='TEST')
        test_saved += a; test_bad += b; test_skipped += c; test_resumed += r

    print("DONE.")
    print("Train windows saved:", train_saved, "resumed:", train_resumed)
    print("Test  windows saved:", test_saved,  "resumed:", test_resumed)
    print("BAD files. Train bad:", train_bad, "Test bad:", test_bad)
    print("SKIPPED.   Train skipped:", train_skipped, "Test skipped:", test_skipped)

if __name__ == '__main__':
    main()