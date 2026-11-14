"""Map the words that OCR read to room kinds, and count them.

An OCR result is a dict with ``W``, ``H`` and ``tokens``. A token is a dict with ``text``,
``conf`` (0 to 1), ``box`` (x0, y0, x1, y1 in image pixels) and optionally ``src``.
Ported from the research prototype; the rules were tuned on 600 Finnish and 400 Swiss plans.
"""
# ruff: noqa
import functools
import re
import unicodedata

# ---------------------------------------------------------------- Finnish lexicon
# Each pattern is matched against a single "word" (split on space, /, +, -, comma, parentheses).
FI = {
    "bedroom": r"(MH\d*|MAKUUH(UONE)?\d*|SOVRUM\d*|SOV\d*)",
    "sauna": r"(S|SAUNA|LH|LÖYLYH(UONE)?|BASTU)",
    "kitchen": r"(K|KK|KT|TK|AVOK|AVOKEITTIÖ|KEITTIÖ|KEITTOK(OMERO)?|APUK|APUKEITTIÖ|TUPAK|TUPAKEITTIÖ|KÖK|KOKVRÅ|KV)",
    "bathroom": r"(KH|KPH|PH|PSH|PESUH(UONE)?|SH|KYLPYH(UONE)?|BADRUM|BAD|DUSCH|TVÄTTRUM|TVÄTT)",
    "wc": r"(WC|W\.C|TOA)",
    "balcony": r"(PARV|PARVEKE|LASITETTU|LASIPARVEKE|LAS\.?PARV|BALKONG|BALK|BALKONGER)",
}
ROOM_WORDS = re.compile(r"^(" + "|".join(FI.values()) + r"|OH|RT|ET|VH|ATK|TH|KHH|KUR|VAR|AK|AH|HH|PUKUH|TEKN|TUULIK|KUISTI|RUOK|\d+[.,]?\d*|M2|M²)$")
WORD_SPLIT = re.compile(r"[\s/+,()\-:;]+")
TYPOLOGY = re.compile(r"\d\s*[HhK]\s*\+|\+\s*[Ss]\s*$")


def norm(s):
    s = unicodedata.normalize("NFC", s).upper()
    # cyrillic/greek lookalikes produced by OCR on rotated text
    tr = str.maketrans({"К": "K", "Р": "P", "Н": "H", "М": "M", "С": "C", "Т": "T", "Е": "E", "О": "O", "А": "A", "В": "B", "Х": "X"})
    return s.translate(tr).strip(" .:;'\"")


def words(text):
    return [w for w in (norm(x) for x in WORD_SPLIT.split(text)) if w]


def raw_words(text):
    return [x.strip(" .:;'\"") for x in WORD_SPLIT.split(text) if x.strip(" .:;'\"")]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def _contain(a, b):
    """fraction of smaller box covered by the intersection"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    sa = (a[2] - a[0]) * (a[3] - a[1]); sb = (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / max(1e-6, min(sa, sb))


FUZZY = {
    "bedroom": ["MAKUUHUONE", "MAKUUH", "SOVRUM"],
    "sauna": ["SAUNA", "LÖYLYHUONE"],
    "kitchen": ["KEITTIÖ", "KEITTOKOMERO", "AVOKEITTIÖ", "TUPAKEITTIÖ", "APUKEITTIÖ"],
    "bathroom": ["PESUHUONE", "KYLPYHUONE", "PESUH", "BADRUM"],
    "wc": [],
    "balcony": ["PARVEKE", "LASITETTU", "BALKONG"],
}


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_word(w, kind):
    for f in FUZZY[kind]:
        tol = 1 if len(f) <= 7 else 2
        if abs(len(w) - len(f)) <= tol and _lev(w, f) <= tol:
            return True
    return False


def token_matches(t, kind, strict=True):
    pat = re.compile(r"^" + FI[kind] + r"$")
    if TYPOLOGY.search(t["text"]):
        return False
    rw = raw_words(t["text"])
    ws = [norm(w) for w in rw]
    ok = False
    for w, r in zip(ws, rw):
        if not pat.match(w) and not (len(w) >= 5 and fuzzy_word(w, kind)):
            continue
        if strict and kind in ("kitchen", "sauna") and len(w) <= 2 and any(c.islower() for c in r):
            continue  # abbreviations are printed in capitals; lowercase 'k', 'tk' = fridge/dishwasher notes
        ok = True
    if not ok:
        return False
    if strict and any(len(w) == 1 and pat.match(w) for w in ws):
        # single letters (K, S) only when every other word is a room word or an area
        if not all(ROOM_WORDS.match(w) for w in ws):
            return False
    return True


def fi_hits(ocr, kind, min_conf=0.3, strict=True):
    """Return clusters of evidence tokens for room labels of `kind`, deduplicated across rotations."""
    hits = [t for t in ocr["tokens"] if t["conf"] >= min_conf and token_matches(t, kind, strict)]
    # bilingual FI/SV plans: drop Swedish words if Finnish ones exist
    sv = re.compile(r"^(SOVRUM\d*|SOV\d*|BASTU|KÖK|KOKVRÅ|BADRUM|BAD|DUSCH|TVÄTTRUM|TVÄTT|BALKONG|BALKONGER|BALK|TOA)$")
    SVW = ["SOVRUM", "BASTU", "BADRUM", "BALKONG", "TVÄTTRUM"]
    is_sv = lambda t: any(sv.match(w) or (len(w) >= 5 and any(_lev(w, x) <= 1 for x in SVW)) for w in words(t["text"]))
    if any(not is_sv(t) for t in hits):
        hits = [t for t in hits if not is_sv(t)]
    clusters = []
    for h in sorted(hits, key=lambda t: -t["conf"]):
        for c in clusters:
            if _contain(c[0]["box"], h["box"]) > 0.5:
                c.append(h); break
        else:
            clusters.append([h])
    return clusters


def fi_wc_separate(ocr, min_conf=0.3):
    """WC clusters that are not part of a combined bathroom label (KPH/WC, KH+WC...)."""
    wcs = fi_hits(ocr, "wc", min_conf)
    baths = fi_hits(ocr, "bathroom", min_conf)
    out = []
    for c in wcs:
        b = c[0]["box"]; h = min(b[3] - b[1], b[2] - b[0])
        if any(re.match("^" + FI["bathroom"] + "$", w) for t in c for w in words(t["text"])):
            continue
        near = False
        for bc in baths:
            bb = bc[0]["box"]
            dx = max(0, max(b[0], bb[0]) - min(b[2], bb[2])); dy = max(0, max(b[1], bb[1]) - min(b[3], bb[3]))
            if dx < 1.0 * h and dy < 1.0 * h:
                near = True
        if not near:
            out.append(c)
    return out


STOREY = re.compile(r"(\d\.?\s*(KERROS|KRS)|KELLARI(KERROS)?|ULLAKKO|POHJAKERROS|YLÄKERTA|ALAKERTA|YLÄKERROS|ALAKERROS|\bPARVI\b|\d\.?\s*VÅN(ING)?|ENTRÉPLAN|KÄLLARE)")


TYPO = re.compile(r"(\d)\s*H\s*((?:\+\s*[A-ZÄÖ/]+\s*)+)")


def typology(ocr):
    """apartment typology string like '3H+K+S' -> dict, or None"""
    for t in ocr["tokens"]:
        m = TYPO.search(norm(t["text"]))
        if m and t["conf"] >= 0.5:
            parts = [p.strip() for p in m.group(2).split("+") if p.strip()]
            return {"rooms": int(m.group(1)), "parts": parts, "token": t}
    return None


def count_cubi(ocr, kind, use_typology=True):
    n, ev = count_cubi_labels(ocr, kind)
    if not use_typology:
        return n, ev
    ty = typology(ocr)
    if ty:
        if kind == "sauna" and n == 0 and "S" in ty["parts"]:
            return 1, [ty["token"]]
        if kind == "kitchen" and n != 1 and any(re.match("^" + FI["kitchen"] + "$", p) for p in ty["parts"]):
            return 1, [ty["token"]] + ev
    return n, ev


def count_cubi_labels(ocr, kind):
    if kind == "storeys":
        heads = set()
        for t in ocr["tokens"]:
            m = STOREY.search(norm(t["text"]))
            if m and t["conf"] >= 0.5:
                heads.add(m.group(0).replace(" ", ""))
        return max(1, len(heads)), []
    if kind == "wc":
        cl = fi_wc_separate(ocr)
    else:
        cl = fi_hits(ocr, kind)
    return len(cl), [c[0] for c in cl]


# ---------------------------------------------------------------- Swiss
LEGEND = re.compile(r"^(Legende|Légende|Legenda)$", re.I)


def plan_cut(ocr):
    ys = [t["box"][1] for t in ocr["tokens"] if LEGEND.match(t["text"].strip())]
    return min(ys) - 5 if ys else 0.82 * ocr["H"]


def strip_acc(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower().replace("’", "'")


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


ROOMNO = re.compile(r"[\s\[\]|]*\d+[.,:]\d+\w?[\s\]|]*$")


SWISS_VOCAB = ['Küche', 'Korridor', 'Bad', 'Zimmer', 'Balkon', 'Treppenhaus', 'Wohnen/Essen', 'Cuisine', 'Couloir',
               'Reduit', 'Bain', 'Wohnen', 'Pièce', 'Schlafzimmer', 'Balcon', "Cage d'escalier", 'Séjour/repas', 'Loggia',
               'Cucina', 'Corridoio', 'Bagno', 'Stanza', 'Vano scala', 'Chambre à coucher', 'Séjour', 'Réduit', 'Balcone',
               'Soggiorno/pranzo', 'Soggiorno', 'Ripostiglio', 'Camera da letto', 'Küche/Essen', 'Kellerabteil',
               'Vano cantina', 'Lift', 'Ascenseur', 'Ascensore', 'Schacht', 'Terrasse', 'Terrazza', 'Dusche', 'Douche',
               'Doccia', 'WC', 'Keller', 'Cave', 'Cantina', 'Abstellraum', 'Waschküche', 'Buanderie', 'Lavanderia']


def approx_sub(pat, text):
    """min edit distance between pat and any substring of text (Sellers)."""
    prev = [0] * (len(text) + 1)
    for i, cp in enumerate(pat, 1):
        cur = [i]
        for j, ct in enumerate(text, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cp != ct)))
        prev = cur
    return min(prev)


def tol(lab):
    n = len(lab)
    return 0 if n <= 3 else 1 if n <= 7 else 2


@functools.lru_cache(maxsize=None)
def _best_label(txt, vocab):
    return best_label_raw(txt, vocab)


def best_label(txt, vocab):
    return _best_label(txt, tuple(sorted(vocab)))


def best_label_raw(txt, vocab):
    """label in vocab best explaining txt; prefer longest label within tolerance."""
    best, bscore = None, -1
    for lab in vocab:
        d = approx_sub(lab, txt)
        if d <= tol(lab):
            sc = len(lab) - d
            if sc > bscore:
                best, bscore = lab, sc
    return best


def swiss_label_hits(ocr, label, fuzzy=True):
    cut = plan_cut(ocr)
    lab = strip_acc(label.strip())
    vocab = {strip_acc(v) for v in SWISS_VOCAB} | {lab}
    pat = re.compile(r"(?<![\w/])" + re.escape(lab) + r"(?![\w/])")
    hits = []
    for t in ocr["tokens"]:
        if t["box"][3] > cut:
            continue
        txt = strip_acc(t["text"]).replace(" / ", "/").strip(" .|[]•-()\"'_,:;°")
        txt = ROOMNO.sub("", txt).strip()
        n_exact = len(pat.findall(txt))
        if n_exact >= 2:
            hits += [t] * n_exact
        elif n_exact == 1 or (fuzzy and len(txt) <= len(lab) + 6 and best_label(txt, vocab) == lab):
            # a 1-char label like 'Bad' needs tight context: reject if it is part of a longer word
            hits.append(t)
    out = []
    for h in hits:
        if any(_contain(o["box"], h["box"]) > 0.5 and o.get("src") != h.get("src") for o in out):
            continue
        out.append(h)
    return out


APT = re.compile(r"\b(WHG|APP|APPT|APT|WOHNUNG|LOGEMENT|APPARTAMENTO)\.?\s*([A-Z]{1,2})\b")


def swiss_apartments(ocr):
    cut = plan_cut(ocr)
    tags = {}
    for t in ocr["tokens"]:
        if t["box"][3] > cut:
            continue
        for m in APT.finditer(norm(t["text"])):
            tags.setdefault(m.group(2), t)
    return list(tags.values())


def swiss_apartments_count(ocr, seq=True):
    tags = swiss_apartments(ocr)
    n = len(tags)
    letters = [m.group(2) for t in tags for m in APT.finditer(norm(t["text"])) if len(m.group(2)) == 1 and m.group(2).isalpha()]
    if seq and letters:
        n = max(n, ord(max(letters)) - ord("A") + 1)
    return n, tags
