"""
Эталонный скрипт расчёта метрик.

Что считается и из какого файла:
  submission.csv   -> mAP@10, Rank-1, Rank-5          (основная метрика, 45%)
  embeddings.npy   -> mAP (полное ранжирование), mINP  (диагностика, справочно)
  candidates.csv   -> Precision, Recall, F1, TNR, PR-AUC (режим отказа, 10%)

Протокол (важные детали, о которых спрашивали):

  1. Junk-пары. Из ранжирования для запроса q удаляются объекты галереи, у
     которых ОДНОВРЕМЕННО тот же vehicle_id И тот же camera_id, что у q.
     Объекты той же камеры с ДРУГИМ vehicle_id остаются в галерее — это
     валидные негативы.
     Фильтр применяется ДО усечения до 10 позиций.

  2. Open-set запросы. Запрос, у которого после junk-фильтрации в галерее
     не осталось ни одного верного ответа, НЕ участвует в mAP/Rank —
     он не получает AP=0, а просто исключается из ranking-метрик.
     Такие запросы оцениваются только в режиме отказа (F1/TNR).

  3. AP@10 нормируется на min(n_pos, 10), где n_pos — число валидных
     позитивов в галерее после junk-фильтрации. Иначе ТС, снятое многими
     камерами, не могло бы получить AP=1 даже при идеальном топ-10.

  4. Ничьи (одинаковые расстояния) разрешаются порядком, в котором объекты
     перечислены участником в submission.csv. При расчёте по embeddings.npy
     ничьи разрешаются порядком строк в test_gallery.csv.

Формат ground truth (есть только у организаторов):
    image_id,vehicle_id,camera_id,split       split in {query, gallery}

Формат submission.csv (без заголовка):
    query_id,gallery_id_1,...,gallery_id_10

Формат candidates.csv (с заголовком):
    query_id,gallery_id,confidence
  Отказ кодируется ОТСУТСТВИЕМ строк для этого query_id.

Запуск:
    python evaluate.py --gt test_ground_truth.csv --submission submission.csv
    python evaluate.py --gt test_ground_truth.csv --submission submission.csv \
        --candidates candidates.csv \
        --embeddings embeddings.npy \
        --query test_query.csv --gallery test_gallery.csv \
        --json report.json
"""

import argparse
import csv
import json
import sys

import numpy as np
import pandas as pd

TOP_K = 10


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------

def load_gt(path):
    gt = pd.read_csv(path, dtype={"image_id": str})
    for col in ("image_id", "vehicle_id", "camera_id", "split"):
        if col not in gt.columns:
            sys.exit(f"В {path} нет обязательной колонки '{col}'")
    query = gt[gt.split == "query"].set_index("image_id")
    gallery = gt[gt.split == "gallery"].set_index("image_id")
    if query.empty or gallery.empty:
        sys.exit("В ground truth пустой query или gallery")
    return query, gallery


def load_submission(path, gallery_ids):
    """
    Читает ранжирование. Возвращает {query_id: [gallery_id, ...]}.
    """
    ranked, unknown, dupes = {}, 0, 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for vals in reader:
            vals = [v.strip() for v in vals if v is not None and v.strip()]
            if not vals:
                continue
            qid, preds = vals[0], vals[1:]

            seen, clean = set(), []
            for p in preds:
                if p not in gallery_ids:
                    unknown += 1
                    continue
                if p in seen:
                    dupes += 1
                    continue
                seen.add(p)
                clean.append(p)
            ranked[qid] = clean

    if unknown:
        print(f"  ! {unknown} gallery_id из submission.csv нет в галерее — отброшены")
    if dupes:
        print(f"  ! {dupes} повторов gallery_id внутри одного запроса — отброшены")
    return ranked


def load_candidates(path):
    """Возвращает {query_id: [(gallery_id, confidence), ...]} по убыванию conf."""
    df = pd.read_csv(path, dtype={"query_id": str, "gallery_id": str})
    for col in ("query_id", "gallery_id", "confidence"):
        if col not in df.columns:
            sys.exit(f"В {path} нет обязательной колонки '{col}'")
    df = df[df.gallery_id.notna() & (df.gallery_id.astype(str).str.strip() != "")]
    out = {}
    for qid, grp in df.groupby("query_id"):
        pairs = sorted(
            ((r.gallery_id, float(r.confidence)) for r in grp.itertuples()),
            key=lambda t: -t[1],
        )
        out[str(qid)] = pairs
    return out


def load_embeddings(path, query_csv, gallery_csv):
    """
    Порядок строк: сначала все объекты test_query.csv в порядке строк файла,
    затем все объекты test_gallery.csv в порядке строк файла.
    """
    emb = np.load(path)
    q_df = pd.read_csv(query_csv, dtype={"image_id": str})
    g_df = pd.read_csv(gallery_csv, dtype={"image_id": str})
    n_q, n_g = len(q_df), len(g_df)

    if emb.ndim != 2:
        sys.exit(f"embeddings.npy должен быть 2D, получено {emb.shape}")
    if emb.shape[0] != n_q + n_g:
        sys.exit(
            f"embeddings.npy: {emb.shape[0]} строк, ожидалось "
            f"{n_q} (query) + {n_g} (gallery) = {n_q + n_g}"
        )

    emb = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)          # L2 на нашей стороне
    return emb[:n_q], emb[n_q:], q_df.image_id.tolist(), g_df.image_id.tolist()


# --------------------------------------------------------------------------
# Ranking-метрики
# --------------------------------------------------------------------------

def valid_positives(query_row, gallery):
    """Число позитивов в галерее после junk-фильтрации (same vid AND same cam)."""
    same_vid = gallery.vehicle_id == query_row.vehicle_id
    same_cam = gallery.camera_id == query_row.camera_id
    return int((same_vid & ~same_cam).sum())


def strip_junk(ranked_list, query_row, gal_vid, gal_cam):
    """Убирает junk (тот же vid и та же камера). Прочие объекты сохраняются."""
    out = []
    for gid in ranked_list:
        if gid not in gal_vid:
            continue
        if gal_vid[gid] == query_row.vehicle_id and gal_cam[gid] == query_row.camera_id:
            continue                                   # junk
        out.append(gid)
    return out


def ranking_metrics(query, gallery, ranked, top_k=TOP_K, ranks=(1, 5)):
    gal_vid = gallery.vehicle_id.to_dict()
    gal_cam = gallery.camera_id.to_dict()

    aps, hits = [], {k: [] for k in ranks}
    n_openset, n_missing = 0, 0

    for qid, row in query.iterrows():
        n_pos = valid_positives(row, gallery)
        if n_pos == 0:
            n_openset += 1                              # open-set: вне mAP/Rank
            continue

        if qid not in ranked:
            n_missing += 1
            aps.append(0.0)
            for k in ranks:
                hits[k].append(False)
            continue

        clean = strip_junk(ranked[qid], row, gal_vid, gal_cam)[:top_k]
        rel = np.array([gal_vid[g] == row.vehicle_id for g in clean], dtype=bool)

        if rel.any():
            cum = np.cumsum(rel)
            prec = cum / (np.arange(len(rel)) + 1)
            aps.append(float((prec * rel).sum() / min(n_pos, top_k)))
        else:
            aps.append(0.0)

        for k in ranks:
            hits[k].append(bool(rel[:k].any()))

    if n_missing:
        print(f"  ! {n_missing} запросов отсутствуют в submission.csv — AP=0")
    return {
        "n_scored": len(aps),
        "n_openset_excluded": n_openset,
        f"mAP@{top_k}": float(np.mean(aps)) if aps else 0.0,
        **{f"Rank-{k}": float(np.mean(hits[k])) if hits[k] else 0.0 for k in ranks},
    }


def full_ranking_metrics(q_emb, g_emb, q_ids, g_ids, query, gallery):
    """mAP по полному ранжированию и mINP. Считается из embeddings.npy."""
    gal_vid = gallery.vehicle_id.to_dict()
    gal_cam = gallery.camera_id.to_dict()

    g_vid = np.array([gal_vid.get(g, -1) for g in g_ids])
    g_cam = np.array([gal_cam.get(g, -1) for g in g_ids])

    sims = q_emb @ g_emb.T
    aps, inps = [], []

    for i, qid in enumerate(q_ids):
        if qid not in query.index:
            continue
        row = query.loc[qid]
        junk = (g_vid == row.vehicle_id) & (g_cam == row.camera_id)
        keep = ~junk
        if not keep.any():
            continue

        order = np.argsort(-sims[i][keep], kind="stable")
        rel = (g_vid[keep] == row.vehicle_id)[order]
        n_pos = int(rel.sum())
        if n_pos == 0:
            continue

        cum = np.cumsum(rel)
        prec = cum / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / n_pos))

        hardest = int(np.max(np.nonzero(rel)[0])) + 1   # 1-based ранг последнего верного
        inps.append(n_pos / hardest)

    return {
        "mAP_full": float(np.mean(aps)) if aps else 0.0,
        "mINP": float(np.mean(inps)) if inps else 0.0,
        "n_scored": len(aps),
    }


# --------------------------------------------------------------------------
# Режим отказа
# --------------------------------------------------------------------------

def candidate_metrics(query, gallery, candidates):
    """
    Решение принимается НА УРОВНЕ ЗАПРОСА, а не пары query-gallery.

      TP  запрос имеет позитив в галерее, команда вернула кандидатов,
          и кандидат с наибольшей confidence — верный
      FP  команда вернула кандидатов, но верхний неверен,
          либо у запроса вообще нет позитива (ложное срабатывание open-set)
      FN  у запроса есть позитив, но команда отказалась отвечать
      TN  у запроса нет позитива, и команда корректно отказалась

      TNR = TN / (TN + FP_openset)  — только по запросам без пары в галерее

    Лишние кандидаты ниже верхнего на F1 не влияют: оценивается решение
    «есть совпадение / нет совпадения», а не каждая пара отдельно.
    """
    gal_vid = gallery.vehicle_id.to_dict()
    tp = fp = fn = tn = 0
    fp_openset = 0
    scores, labels = [], []

    for qid, row in query.iterrows():
        has_match = valid_positives(row, gallery) > 0
        returned = candidates.get(qid, [])

        labels.append(1 if has_match else 0)
        scores.append(returned[0][1] if returned else float("-inf"))

        if returned:
            top_gid = returned[0][0]
            correct = gal_vid.get(top_gid, None) == row.vehicle_id
            if has_match and correct:
                tp += 1
            else:
                fp += 1
                if not has_match:
                    fp_openset += 1
        else:
            if has_match:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tnr = tn / (tn + fp_openset) if (tn + fp_openset) else float("nan")

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "n_openset_queries": tn + fp_openset,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TNR": tnr,
        "PR-AUC": pr_auc(np.array(scores), np.array(labels)),
    }


def pr_auc(scores, labels):
    """
    Average Precision по кривой Precision-Recall для решения
    «у запроса есть пара в галерее». Порогонезависимая мера качества
    самого скора уверенности; не зависит от выбранного командой порога.
    """
    finite = np.isfinite(scores)
    if labels.sum() == 0 or not finite.any():
        return float("nan")

    s = np.where(finite, scores, np.min(scores[finite]) - 1.0)
    order = np.argsort(-s, kind="stable")
    y = labels[order]

    cum_tp = np.cumsum(y)
    prec = cum_tp / (np.arange(len(y)) + 1)
    rec = cum_tp / labels.sum()

    ap, prev_rec = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_rec)
        prev_rec = r
    return float(ap)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Эталонный расчёт метрик, задача 7")
    ap.add_argument("--gt", required=True, help="test_ground_truth.csv")
    ap.add_argument("--submission", required=True, help="submission.csv")
    ap.add_argument("--candidates", default=None, help="candidates.csv")
    ap.add_argument("--embeddings", default=None, help="embeddings.npy")
    ap.add_argument("--query", default=None, help="test_query.csv (для --embeddings)")
    ap.add_argument("--gallery", default=None, help="test_gallery.csv (для --embeddings)")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--json", default=None, help="куда сохранить отчёт")
    args = ap.parse_args()

    query, gallery = load_gt(args.gt)
    print(f"ground truth   : {len(query)} query, {len(gallery)} gallery, "
          f"{gallery.vehicle_id.nunique()} ТС в галерее")

    report = {}

    print("\n--- Ранжирование (submission.csv) ---")
    ranked = load_submission(args.submission, set(gallery.index))
    rm = ranking_metrics(query, gallery, ranked, top_k=args.top_k)
    report["ranking"] = rm
    print(f"запросов в зачёте : {rm['n_scored']}")
    if rm["n_openset_excluded"]:
        print(f"open-set вне mAP  : {rm['n_openset_excluded']}")
    print(f"mAP@{args.top_k}           : {rm[f'mAP@{args.top_k}']:.4f}   <- основная метрика")
    print(f"Rank-1            : {rm['Rank-1']:.4f}")
    print(f"Rank-5            : {rm['Rank-5']:.4f}")

    if args.embeddings:
        if not (args.query and args.gallery):
            sys.exit("--embeddings требует --query и --gallery")
        print("\n--- Полное ранжирование (embeddings.npy, справочно) ---")
        q_emb, g_emb, q_ids, g_ids = load_embeddings(
            args.embeddings, args.query, args.gallery)
        fm = full_ranking_metrics(q_emb, g_emb, q_ids, g_ids, query, gallery)
        report["full_ranking"] = fm
        print(f"mAP (полный)      : {fm['mAP_full']:.4f}")
        print(f"mINP              : {fm['mINP']:.4f}")

    if args.candidates:
        print("\n--- Режим предложения кандидатов (candidates.csv) ---")
        cands = load_candidates(args.candidates)
        cm = candidate_metrics(query, gallery, cands)
        report["candidates"] = cm
        print(f"TP/FP/FN/TN       : {cm['TP']}/{cm['FP']}/{cm['FN']}/{cm['TN']}")
        print(f"Precision         : {cm['Precision']:.4f}")
        print(f"Recall            : {cm['Recall']:.4f}")
        print(f"F1                : {cm['F1']:.4f}")
        if cm["n_openset_queries"]:
            print(f"TNR               : {cm['TNR']:.4f}  "
                  f"({cm['n_openset_queries']} запросов без пары)")
        else:
            print("TNR               : n/a — в выборке нет запросов без пары в галерее")
        print(f"PR-AUC            : {cm['PR-AUC']:.4f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nотчёт -> {args.json}")


if __name__ == "__main__":
    main()