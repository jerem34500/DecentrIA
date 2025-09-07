#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import math
import time
import shutil
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ===========================
# Zero-shot (DeBERTa MNLI)
# ===========================
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ----------- Utils I/O -----------
def read_text(path: Path, max_chars: int = 200_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read(max_chars)
        # nettoyage très léger
        txt = txt.replace("\r", " ").replace("\t", " ")
        return txt
    except Exception as e:
        return ""


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move_file(src: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    try:
        if dst.exists():
            # éviter collisions
            stem = dst.stem
            suf = dst.suffix
            i = 1
            while True:
                cand = dst_dir / f"{stem}__{i}{suf}"
                if not cand.exists():
                    dst = cand
                    break
                i += 1
        shutil.move(str(src), str(dst))
    except Exception:
        # dernier recours : copie+suppression
        try:
            shutil.copy2(str(src), str(dst))
            src.unlink(missing_ok=True)
        except Exception:
            pass


# ----------- Chargement catégories -----------
def load_categories(cats_py_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Accepte un module Python qui contient:
      - soit CATEGORIES = {"Culture": {"Musique": [...], "Cinema": [...]}, ...}
      - soit CAT_KEYWORDS = même forme mais on ignore les listes de mots
    Renvoie:
      - liste des catégories
      - dict catégorie -> liste de sous-catégories
    """
    p = Path(cats_py_path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier catégories introuvable: {cats_py_path}")

    spec = importlib.util.spec_from_file_location("cats_mod", cats_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    if hasattr(mod, "CATEGORIES"):
        src = getattr(mod, "CATEGORIES")
    elif hasattr(mod, "CAT_KEYWORDS"):
        src = getattr(mod, "CAT_KEYWORDS")
    else:
        raise ValueError("Le module catégories doit définir CATEGORIES ou CAT_KEYWORDS.")

    cats = []
    sub = {}
    for cat, subdict in src.items():
        cats.append(cat)
        # subdict est un dict "sous-cat" -> [keywords...] ; on garde juste les noms
        sub[cat] = list(subdict.keys())
    return cats, sub


# ----------- Construction pipeline ZS -----------
def build_zero_shot_pipeline(model_name: str, device: Optional[int] = None):
    """
    DeBERTa-v3 Large MNLI par défaut.
    Si indisponible, on bascule automatiquement sur BART MNLI.
    """
    tried = []
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        nli = pipeline(
            "zero-shot-classification",
            model=mdl,
            tokenizer=tok,
            device=device if device is not None else -1,
        )
        return nli, tok
    except Exception as e:
        tried.append((model_name, str(e)))

    # fallback robuste
    fallback = "facebook/bart-large-mnli"
    try:
        tok = AutoTokenizer.from_pretrained(fallback)
        mdl = AutoModelForSequenceClassification.from_pretrained(fallback)
        nli = pipeline(
            "zero-shot-classification",
            model=mdl,
            tokenizer=tok,
            device=device if device is not None else -1,
        )
        print(
            f"[INFO] Modèle '{model_name}' indisponible. "
            f"Bascule vers fallback '{fallback}'.",
            file=sys.stderr,
        )
        return nli, tok
    except Exception as e2:
        tried.append((fallback, str(e2)))
        msgs = "\n".join([f"- {m} -> {err}" for m, err in tried])
        raise RuntimeError(
            "Impossible de charger un modèle zero-shot.\n" + msgs
        )


# ----------- Découpage token -----------
def tokenize_chunks(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = 512,
    stride: int = 64,
    max_chunks: int = 8,
) -> List[str]:
    """
    Découpage par tokens avec chevauchement (stride).
    On re-décode chaque tranche pour envoyer de « vraies phrases » au pipeline.
    """
    if not text:
        return []
    enc = tokenizer(
        text,
        truncation=False,
        padding=False,
        return_tensors=None,
        return_offsets_mapping=False,
        add_special_tokens=False,
    )
    ids = enc["input_ids"]
    if len(ids) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(ids) and len(chunks) < max_chunks:
        end = min(start + max_tokens, len(ids))
        piece_ids = ids[start:end]
        chunk = tokenizer.decode(piece_ids, skip_special_tokens=True)
        chunks.append(chunk)
        if end == len(ids):
            break
        start = end - stride  # chevauchement
        if start < 0:
            start = 0
    return chunks


# ----------- Agrégation scores -----------
def aggregate_scores_per_label(all_results: List[Dict]) -> Dict[str, float]:
    """
    all_results : liste de sorties du pipeline pour chaque chunk:
      {"labels":[...], "scores":[...]} (softmax)
    Agrège par moyenne des meilleurs scores par label, pondération « max-par-chunk ».
    """
    labels = set()
    for r in all_results:
        labels.update(r["labels"])
    labels = list(labels)

    # dict label -> liste des scores par chunk (on prend le score correspondant)
    bag = {lab: [] for lab in labels}
    for r in all_results:
        m = {lab: sc for lab, sc in zip(r["labels"], r["scores"])}
        for lab in labels:
            if lab in m:
                bag[lab].append(m[lab])

    # agrégation: moyenne des meilleurs k par label
    agg = {}
    for lab, scores in bag.items():
        if not scores:
            agg[lab] = 0.0
        else:
            scores_sorted = sorted(scores, reverse=True)
            k = min(3, len(scores_sorted))
            agg[lab] = sum(scores_sorted[:k]) / k
    return agg


# ----------- Classification A+B -----------
def classify_text_2steps(
    text: str,
    nli_pipe,
    tokenizer,
    categories: List[str],
    subcats: Dict[str, List[str]],
    *,
    hyp_cat: str = "This text is about {}.",
    hyp_sub: str = "This text is about {}.",
    max_tokens: int = 512,
    stride: int = 64,
    max_chunks: int = 8,
    cat_thresh: float = 0.40,
    sub_thresh: float = 0.45,
) -> Tuple[str, str, float, float, str]:
    """
    Retourne: (cat, sub, cat_score, sub_score, decision)
    decision ∈ {"ASSIGNED", "NEEDS_REVIEW", "UNASSIGNED"}
    """
    chunks = tokenize_chunks(text, tokenizer, max_tokens, stride, max_chunks)
    if not chunks:
        return "", "", 0.0, 0.0, "UNASSIGNED"

    # Étape A : Catégorie
    cat_results = []
    for ch in chunks:
        r = nli_pipe(
            ch,
            candidate_labels=categories,
            hypothesis_template=hyp_cat,
            multi_label=False,
        )
        # normalise vers dict
        cat_results.append({"labels": r["labels"], "scores": r["scores"]})
    agg_cat = aggregate_scores_per_label(cat_results)

    best_cat = max(agg_cat.items(), key=lambda x: x[1])
    cat_name, cat_score = best_cat
    if cat_score < cat_thresh:
        return "", "", cat_score, 0.0, "UNASSIGNED"

    # Étape B : Sous-catégories de la gagnante
    subs = subcats.get(cat_name, [])
    if not subs:
        # pas de sous-catégories → assignation directe à la catégorie seule
        return cat_name, "", cat_score, 0.0, "ASSIGNED"

    sub_results = []
    for ch in chunks:
        r = nli_pipe(
            ch,
            candidate_labels=subs,
            hypothesis_template=hyp_sub,
            multi_label=False,
        )
        sub_results.append({"labels": r["labels"], "scores": r["scores"]})
    agg_sub = aggregate_scores_per_label(sub_results)
    best_sub = max(agg_sub.items(), key=lambda x: x[1])
    sub_name, sub_score = best_sub

    if sub_score >= sub_thresh:
        return cat_name, sub_name, cat_score, sub_score, "ASSIGNED"
    else:
        return cat_name, sub_name, cat_score, sub_score, "NEEDS_REVIEW"


# ----------- CLI / Main -----------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Classement sémantique hiérarchique (Zéro‑shot MNLI, pas de mots‑clé)."
    )
    ap.add_argument("--cats", required=True, help="Chemin du module catégories .py")
    ap.add_argument("--in_dir", required=True, help="Dossier d'entrée (out_clean)")
    ap.add_argument("--out_root", required=True, help="Racine des dossiers classés")
    ap.add_argument("--review_dir", required=True, help="Dossier des NEEDS_REVIEW")
    ap.add_argument("--report", required=True, help="CSV de rapport")
    ap.add_argument(
        "--model",
        default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        help="Modèle HF (MNLI) — fallback auto vers facebook/bart-large-mnli si indispo.",
    )
    ap.add_argument("--device", type=int, default=None, help="GPU id (ex: 0) ou None")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--max_chunks", type=int, default=8)
    ap.add_argument("--min_len", type=int, default=400, help="Longueur min (caractères)")
    ap.add_argument("--cat_thresh", type=float, default=0.40)
    ap.add_argument("--sub_thresh", type=float, default=0.45)
    ap.add_argument("--sample", type=int, default=0, help="Limiter le nb de fichiers (debug)")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_root)
    review_dir = Path(args.review_dir)
    report_csv = Path(args.report)

    ensure_dir(out_root)
    ensure_dir(review_dir)
    ensure_dir(report_csv.parent)

    print(">> Chargement catégories…")
    cats, subcats = load_categories(args.cats)
    if not cats:
        print("[ERREUR] Aucune catégorie détectée.", file=sys.stderr)
        sys.exit(1)

    # pipeline + tokenizer
    print(">> Initialisation du modèle…")
    nli_pipe, tokenizer = build_zero_shot_pipeline(args.model, device=args.device)

    files = sorted(
        [p for p in in_dir.glob("**/*") if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    if args.sample and args.sample > 0:
        files = files[: args.sample]

    total = len(files)
    if total == 0:
        print("[INFO] Aucun fichier à traiter.")
        sys.exit(0)

    # Rapport CSV
    f_csv = open(report_csv, "w", newline="", encoding="utf-8")
    cw = csv.writer(f_csv, delimiter=";")
    cw.writerow(
        [
            "file",
            "decision",
            "category",
            "subcategory",
            "cat_score",
            "sub_score",
            "note",
        ]
    )

    n_assigned = 0
    n_review = 0
    n_unassigned = 0
    t0 = time.time()

    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="Classification", unit="file")

    for fp in iterator:
        txt = read_text(fp)
        if len(txt) < args.min_len:
            # trop court → à revoir
            cw.writerow([str(fp), "NEEDS_REVIEW", "", "", 0, 0, "too_short"])
            move_file(fp, review_dir)
            n_review += 1
            continue

        cat, sub, cs, ss, decision = classify_text_2steps(
            txt,
            nli_pipe,
            tokenizer,
            cats,
            subcats,
            hyp_cat="This text is about {}.",
            hyp_sub="This text is about {}.",
            max_tokens=args.max_tokens,
            stride=args.stride,
            max_chunks=args.max_chunks,
            cat_thresh=args.cat_thresh,
            sub_thresh=args.sub_thresh,
        )

        if decision == "ASSIGNED":
            # destination = out_root/cat/subcat/
            dst = out_root / (cat if cat else "UNKNOWN") / (sub if sub else "_")
            move_file(fp, dst)
            n_assigned += 1
            cw.writerow([str(fp), "ASSIGNED", cat, sub, f"{cs:.4f}", f"{ss:.4f}", ""])
        elif decision == "NEEDS_REVIEW":
            move_file(fp, review_dir)
            n_review += 1
            cw.writerow([str(fp), "NEEDS_REVIEW", cat, sub, f"{cs:.4f}", f"{ss:.4f}", "low_subconfidence"])
        else:
            # UNASSIGNED : laisser dans out_clean
            n_unassigned += 1
            cw.writerow([str(fp), "UNASSIGNED", "", "", f"{cs:.4f}", f"{ss:.4f}", "low_catconfidence"])

    f_csv.close()
    dt = time.time() - t0

    print("\n== RÉSUMÉ ==")
    print(f"Fichiers traités : {total} en {dt:.1f}s  (~{total/max(1,dt):.1f}/s)")
    print(f"- ASSIGNED     : {n_assigned}")
    print(f"- NEEDS_REVIEW : {n_review}")
    print(f"- UNASSIGNED   : {n_unassigned}")
    print(f"OK → Rapport : {report_csv}")
    print(f"Sorties       : {out_root} ; {review_dir}")


if __name__ == "__main__":
    main()

Ajout du script de classification RAG
