def match_tracks(pred_tracks, true_tracks):
    """
    Возвращает список сопоставлений:
    (pred_track, best_true_track, purity, efficiency)
    """
    results = []

    true_sets = [set(t) for t in true_tracks]

    for p in pred_tracks:
        p_set = set(p)

        best_match = None
        best_overlap = 0
        best_purity = 0.0
        best_efficiency = 0.0

        for t_set in true_sets:
            overlap = len(p_set & t_set)
            if overlap == 0:
                continue

            purity = overlap / len(p_set)
            efficiency = overlap / len(t_set)

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = t_set
                best_purity = purity
                best_efficiency = efficiency

        results.append(
            {
                "pred_track": p,
                "matched": best_match is not None,
                "purity": best_purity,
                "efficiency": best_efficiency,
                "overlap": best_overlap,
            }
        )

    return results


def track_level_metrics(
    pred_tracks, true_tracks, purity_thr=0.8, efficiency_thr=0.8, min_length=3
):

    matches = match_tracks(pred_tracks, true_tracks)

    good_tracks = 0
    for m in matches:
        if (
            m["matched"]
            and m["purity"] >= purity_thr
            and m["efficiency"] >= efficiency_thr
            and len(m["pred_track"]) >= min_length
        ):
            good_tracks += 1

    return {
        "n_pred": len(pred_tracks),
        "n_true": len(true_tracks),
        "n_good": good_tracks,
        "track_precision": good_tracks / max(len(pred_tracks), 1),
        "track_recall": good_tracks / max(len(true_tracks), 1),
    }


