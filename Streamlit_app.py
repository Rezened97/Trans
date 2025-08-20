# streamlit_app.py

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple, Optional

import requests
import streamlit as st

# =========================
# Utility: rerun compat
# =========================
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # fallback per Streamlit < ~1.30
        st.experimental_rerun()

# =========================
# Auth
# =========================
def require_auth() -> bool:
    # Permetti di disabilitare l'auth (utile in locale) via secrets o env
    if bool(st.secrets.get("DISABLE_AUTH", False)) or os.environ.get("DISABLE_AUTH", "").lower() in {"1", "true", "yes"}:
        return True

    app_pw = st.secrets.get("APP_PASSWORD") or os.environ.get("APP_PASSWORD", "")
    if not app_pw:
        st.error("Autenticazione attiva ma APP_PASSWORD non è impostata (secrets o variabile d'ambiente).")
        st.stop()

    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return True

    st.title("🔒 Accesso richiesto")
    with st.form("login", clear_on_submit=False):
        pwd = st.text_input("Password", type="password")
        ok = st.form_submit_button("Accedi")

    if ok:
        if pwd == app_pw:
            st.session_state.auth_ok = True
            _rerun()
        else:
            st.error("Password errata.")
            st.stop()

    st.stop()

# =========================
# Settings persistence
# =========================
SETTINGS_PATH = os.environ.get("TRANSLATOR_SETTINGS_PATH", ".settings.json")

def load_settings() -> Dict[str, Any]:
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_settings(s: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        return True, f"Impostazioni salvate in {SETTINGS_PATH}"
    except Exception as e:
        return False, f"Impossibile salvare le impostazioni: {e}"

def get_api_key_from_settings_or_env() -> str:
    # Priorità: settings → secrets → env
    s = load_settings()
    if s.get("DEEPL_API_KEY"):
        return s["DEEPL_API_KEY"]
    return st.secrets.get("DEEPL_API_KEY", os.environ.get("DEEPL_API_KEY", ""))

# =========================
# Heuristics & translation core
# =========================

# Toggle diagnostica (disattivata per UI web)
VERBOSE_SKIPS = False
VERBOSE_BATCH = False

# ——— Whitelist base ———
KEY_WHITELIST = {
    "title", "subtitle", "heading", "subheading", "headline",
    "text", "label", "caption", "description", "content",
    "paragraph", "paragraphs", "cta", "button", "button_text",
    "link_text", "alt", "placeholder", "aria-label",
    "q", "a", "question", "answer",
    "meta_description", "og:description", "twitter:description",
    "editor", "tab_title", "tab_content", "inner_text",
    "hero", "offer", "availability", "status",
    "social_proof", "cta_banner", "footer", "misc", "faq_header"
}

# Chiavi testuali tipiche dei form Elementor (aggiunte alla whitelist)
FORM_TEXT_KEYS = {
    "field_label", "placeholder", "help", "description",
    "step_next_label", "step_previous_label",
    "button_text",
    "success_message", "error_message", "server_message",
    "invalid_message", "required_field_message", "required_message",
    "empty_field_message", "email_invalid_message", "upload_invalid_message",
    "acceptance_text"
}
KEY_WHITELIST = KEY_WHITELIST.union(FORM_TEXT_KEYS)

# Chiavi tecniche da NON tradurre mai
KEY_BLOCKLIST = {
    "id", "class", "classes", "selector", "type", "tag",
    "href", "src", "url", "slug", "key", "name_attr",
    "width", "height", "style", "css", "html", "js", "json",
    "color", "background", "font", "dataset", "data", "value_raw",
    # Tecniche dei form (non testuali)
    "form_name", "custom_id", "field_type", "required",
    "options", "value", "values",
    "webhook", "webhooks", "webhooks_advanced_data", "submit_actions",
    "email_content", "email_content_2"  # spesso contiene placeholder come "[all-fields]"
}

# ====== Pattern base ======
RE_URL = re.compile(r"^(https?:)?//|^[\w\-.]+@\w", re.IGNORECASE)
RE_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
RE_MEASURE = re.compile(r"^\d+(\.\d+)?(px|em|rem|vh|vw|%|cm|mm|in)$", re.IGNORECASE)
RE_BASE64 = re.compile(r"^data:[\w/+.-]+;base64,", re.IGNORECASE)
RE_SELECTOR = re.compile(r"^[.#][A-Za-z0-9_\-:.#> \[\]=]+$")
RE_NUMERICISH = re.compile(r"^[\d\s.,:/+\-%°C°F]+$")
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_SINGLE_TOKEN_ASCII = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")

CODE_KEYWORDS = (
    "function", "var ", "let ", "const ", "return", "=>", "if(", "else",
    "{", "}", "[", "]", "(", ")", ";", "/*", "*/", "</", "/>", "document.",
    "window.", "class ", "import ", "export ", "new ", "===", "!==",
    "try", "catch", "finally", "json", "null", "true", "false", "async", "await", ":"
)

def is_html_like(s: str) -> bool:
    return bool(RE_HTML_TAG.search(s))

def html_seems_broken(s: str) -> bool:
    if s.count("<") != s.count(">"):
        return True
    if re.search(r"<[a-zA-Z][^>]*$", s):
        return True
    return False

def fix_bold_spacing(s: str) -> str:
    # Inserisce spazio se manca prima/dopo tag <b>/<strong>
    s = re.sub(r"([^\s>])(<\s*(?:b|strong)\b)", r"\1 \2", s)
    s = re.sub(r"(</\s*(?:b|strong)\s*>)\s*([^\s<])", r"\1 \2", s)
    return s

def looks_like_code_or_nonhuman(s: str) -> Tuple[bool, str]:
    t = s.strip()
    if not t:
        return True, "empty"
    if RE_URL.search(t): return True, "url/email"
    if RE_COLOR.match(t): return True, "color"
    if RE_MEASURE.match(t): return True, "measure"
    if RE_BASE64.match(t): return True, "base64"
    if RE_SELECTOR.match(t): return True, "css_selector"
    if RE_NUMERICISH.match(t) and len(t) <= 12:
        return True, "numericish_only"

    is_html = is_html_like(t)
    if not is_html:
        weird_chars = sum(t.count(ch) for ch in "{}[]();=$|")
        if weird_chars >= max(6, len(t) // 8):
            return True, "too_many_code_chars"
        lower = t.lower()
        if any(kw in lower for kw in CODE_KEYWORDS):
            return True, "code_keywords"

    if len(t) <= 2:
        return True, "too_short"

    # Non marchiamo automaticamente slug qui: lo gestiamo in base alla chiave
    return False, ""

# ====== Regole anti-rottura Elementor ======
# Valori da NON tradurre per chiavi tecniche/enum
ENUM_KEYS_EXACT = {
    "widgetType", "elType", "header_size", "image_size",
    "display_percentage", "text_stroke_text_stroke_type",
    "_element_width", "_element_width_mobile",
}

# Sottostringhe indicative di enum/slug/valori tecnici
ENUM_KEY_SUBSTR = [
    "align", "background_", "_background", "overlay",
    "typography_", "_typography", "font_", "letter_spacing", "line_height",
    "display_", "border", "radius", "icon_align", "justify",
    "object_", "position", "repeat", "attachment", "blend",
    "_element_", "contactform_", "pa_condition_", "__globals__", "__dynamic__",
    "image_size", "header_size", "style",
    # indicatori tecnici utili ma non aggressivi
    "form_", "validate", "validation_", "webhook"
]
# (NOTA: rimosso 'field_', 'step_', 'button_', 'message_' per non bloccare testi dei form)

def is_enumish_key(key: str) -> bool:
    k = (key or "")
    kl = k.lower()
    if k in ENUM_KEYS_EXACT:
        return True
    if any(sub in kl for sub in ENUM_KEY_SUBSTR):
        return True
    if kl.startswith("_"):
        return True
    return False

def likely_human_text(key: str, value: str) -> Tuple[bool, str]:
    """
    True => traducibile; False => lascia invariato.
    Ordine FIX:
      1) Blocklist (mai tradurre)
      2) **Whitelist (sempre tradurre se testo umano)**
      3) Enumish (in caso di dubbio, NON tradurre)
      4) Euristica sul contenuto
    """
    k = (key or "").lower()

    # 1) Mai tradurre
    if k in (kk.lower() for kk in KEY_BLOCKLIST):
        return False, "blocked_key"

    # 2) Se è in whitelist → traducibile (salvo sia evidentemente non-umano)
    if k in (kk.lower() for kk in KEY_WHITELIST):
        is_nonhuman, why = looks_like_code_or_nonhuman(value)
        return (not is_nonhuman, f"whitelisted_but_{why}") if is_nonhuman else (True, "")

    # 3) Se sembra enum/tecnico → NON tradurre
    if is_enumish_key(key):
        return False, "enum_like_key"

    # 4) Heuristica sui contenuti
    letters = sum(ch.isalpha() for ch in value)
    spaces = value.count(" ")
    has_punct = any(p in value for p in [".", "!", "?", ";", ":", ",", "…", "—"])

    if spaces == 0:
        if RE_SINGLE_TOKEN_ASCII.match(value.strip()):
            return False, "single_token_enum"
        is_nonhuman, why = looks_like_code_or_nonhuman(value)
        return (not is_nonhuman, f"single_token_{why}") if is_nonhuman else (False, "single_token_unsure")

    if letters >= 2 and (spaces >= 1 or has_punct):
        is_nonhuman, why = looks_like_code_or_nonhuman(value)
        return (not is_nonhuman, f"heuristic_{why}") if is_nonhuman else (True, "")

    return False, "no_heuristic_match"

def normalize_api_url(api_key: str) -> str:
    # Auto: Pro (default) o Free se la chiave termina con :fx
    return "https://api-free.deepl.com" if api_key.strip().endswith(":fx") else "https://api.deepl.com"

def deepl_batch_translate(
    texts: List[str],
    target_lang: str,
    api_url: str,
    api_key: str,
    session: Optional[requests.Session] = None,
    tag_handling: Optional[str] = None
) -> List[str]:
    if not texts:
        return []

    url = f"{api_url}/v2/translate"
    data = [
        ('target_lang', target_lang.upper()),
        ('preserve_formatting', '1'),
    ]
    if tag_handling:
        data.append(('tag_handling', tag_handling))

    for t in texts:
        data.append(('text', t))

    headers = {'Authorization': f'DeepL-Auth-Key {api_key}'}
    sess = session or requests.Session()

    for attempt in range(4):
        try:
            resp = sess.post(url, data=data, headers=headers, timeout=40)
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    "Autorizzazione rifiutata da DeepL (401/403). Controlla la chiave e il piano (Free/Pro)."
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.2 * (attempt + 1))
                continue

            resp.raise_for_status()
            j = resp.json()
            out = [item['text'] for item in j.get('translations', [])]
            if len(out) != len(texts):
                return texts
            return out
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.2 * (attempt + 1))
    return texts

# ——— Supporto specifico: opzioni dei form come lista di dict ———
def _form_options_label_overrides(node: Any, path: List[Any], acc: List[Dict[str, Any]]) -> bool:
    """
    options come lista di dict: [{'label': 'Sì', 'value': 'yes'}, ...]
    → aggiunge solo i 'label' all'accumulatore per la traduzione.
    Ritorna True se gestito, False altrimenti.
    """
    if isinstance(node, list) and all(isinstance(it, dict) for it in node):
        for i, it in enumerate(node):
            if "label" in it and isinstance(it["label"], str):
                acc.append({"path": path + [i, "label"], "key": "label", "value": it["label"]})
        return True
    return False

def gather_strings(obj: Any, parent_key: str = "") -> List[Dict[str, Any]]:
    acc: List[Dict[str, Any]] = []

    def _walk(node, path, current_key):
        if isinstance(node, dict):
            for k, v in node.items():
                # Se siamo dentro campi di un form Elementor, intercetta options strutturate
                if k == "options" and any(p == "form_fields" for p in path):
                    if _form_options_label_overrides(v, path + [k], acc):
                        continue
                _walk(v, path + [k], k)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, path + [i], current_key)
        elif isinstance(node, str):
            ok, reason = likely_human_text(current_key, node)
            if ok:
                acc.append({"path": path, "key": current_key, "value": node})
            else:
                if VERBOSE_SKIPS:
                    path_str = "/".join(str(p) for p in path)
                    preview = node[:80].replace("\n", "\\n")
                    print(f"[SKIP] path='{path_str}' key='{current_key}' value='{preview}{'...' if len(node)>80 else ''}' reason={reason}")

    _walk(obj, [], parent_key)
    return acc

def set_by_path(obj: Any, path: List[Any], value: Any):
    cur = obj
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value

def translate_all_human_text(
    obj: Any,
    target_lang: str,
    api_key: str,
    progress_cb: Optional[Any] = None
) -> Any:
    api_url = normalize_api_url(api_key)
    refs = gather_strings(obj)
    if not refs:
        return obj

    html_ok: List[Dict[str, Any]] = []
    html_broken_or_minimal: List[Dict[str, Any]] = []
    plain: List[Dict[str, Any]] = []

    for r in refs:
        v = r["value"]
        if is_html_like(v):
            if html_seems_broken(v):
                html_broken_or_minimal.append(r)
            else:
                html_ok.append(r)
        else:
            plain.append(r)

    # batching
    sess = requests.Session()
    MAX_BATCH = 120

    total_chunks = (
        ((len(plain) + MAX_BATCH - 1) // MAX_BATCH) +
        ((len(html_ok) + MAX_BATCH - 1) // MAX_BATCH) +
        ((len(html_broken_or_minimal) + MAX_BATCH - 1) // MAX_BATCH)
    )
    done_chunks = 0

    def process_bucket(bucket: List[Dict[str, Any]], tag: Optional[str]) -> List[str]:
        nonlocal done_chunks
        out: List[str] = []
        for i in range(0, len(bucket), MAX_BATCH):
            chunk = bucket[i:i + MAX_BATCH]
            texts = [x["value"] for x in chunk]
            try:
                trans = deepl_batch_translate(texts, target_lang, api_url, api_key, session=sess, tag_handling=tag)
            except Exception:
                if tag == 'html':
                    try:
                        trans = deepl_batch_translate(texts, target_lang, api_url, api_key, session=sess, tag_handling=None)
                    except Exception:
                        trans = texts
                else:
                    trans = texts
            fixed = []
            for original, t in zip(texts, trans):
                if is_html_like(original) or is_html_like(t):
                    t = fix_bold_spacing(t)
                fixed.append(t)
            out.extend(fixed)

            done_chunks += 1
            if progress_cb:
                progress_cb(done_chunks, total_chunks)
        return out

    plain_out = process_bucket(plain, None)
    html_ok_out = process_bucket(html_ok, 'html')
    html_br_out = process_bucket(html_broken_or_minimal, None)

    for bucket, outs in ((plain, plain_out), (html_ok, html_ok_out), (html_broken_or_minimal, html_br_out)):
        for ref, new_text in zip(bucket, outs):
            set_by_path(obj, ref["path"], new_text)

    return obj

# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="JSON Translator (DeepL)", page_icon="🌐", layout="centered")

    if not require_auth():
        return

    if "logs" not in st.session_state:
        st.session_state.logs = []

    # Persistenza risultati tra i rerun per i download multipli
    if "translation_results" not in st.session_state:
        st.session_state.translation_results: Dict[str, bytes] = {}

    def log(msg: str):
        st.session_state.logs.append(msg)

    st.title("🌐 JSON Translator (DeepL)")

    tabs = st.tabs(["Traduzione", "Impostazioni", "Log"])

    # --------- TAB: Impostazioni ---------
    with tabs[1]:
        st.subheader("Impostazioni")
        st.caption("La chiave viene salvata in un file locale `.settings.json` (se possibile). In alternativa puoi usare `st.secrets` o variabili d'ambiente.")

        current_settings = load_settings()
        cur_key_masked = "••••••••" if current_settings.get("DEEPL_API_KEY") else "(non impostata)"
        st.write(f"Stato chiave salvata: **{cur_key_masked}**")

        with st.form("settings_form", clear_on_submit=False):
            api_key_in = st.text_input("DeepL API Key", type="password", value=current_settings.get("DEEPL_API_KEY", ""))
            save_btn = st.form_submit_button("Salva impostazioni")

        if save_btn:
            new_settings = dict(current_settings)
            new_settings["DEEPL_API_KEY"] = api_key_in.strip()
            ok, msg = save_settings(new_settings)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            _rerun()

        st.divider()
        st.caption("Suggerimento: su Streamlit Cloud/Sharing imposta `APP_PASSWORD` e `DEEPL_API_KEY` in **Secrets**.")

    # --------- TAB: Traduzione ---------
    with tabs[0]:
        st.subheader("Traduci JSON")
        api_key = get_api_key_from_settings_or_env()
        if not api_key:
            st.error("Nessuna DeepL API Key trovata. Vai su **Impostazioni** e salvala.")
            st.stop()

        uploaded = st.file_uploader("Carica il file JSON", type=["json"], accept_multiple_files=False, key="uploader")
        langs = st.text_input("Lingue target (es. EN, FR, HU) — separa con virgole", value="EN")
        langs_list = [x.strip().upper() for x in langs.split(",") if x.strip()]
        colA, colB = st.columns([1,1], gap="small")
        with colA:
            start = st.button("Traduci", use_container_width=True)
        with colB:
            clear_btn = st.button("Pulisci risultati", use_container_width=True)

        if clear_btn:
            st.session_state.translation_results.clear()
            st.success("Risultati precedenti rimossi.")

        # Mostra sempre i risultati presenti in sessione (anche dopo i click di download)
        if st.session_state.translation_results:
            st.success("Traduzione completata. Scarica i file:")
            for fname, content in st.session_state.translation_results.items():
                st.download_button(
                    label=f"⬇️ Scarica {fname}",
                    data=content,
                    file_name=fname,
                    mime="application/json",
                    use_container_width=True,
                    key=f"dl_{fname}"  # chiave stabile → i pulsanti non scompaiono
                )

        if start:
            if not uploaded:
                st.warning("Carica un file JSON prima di avviare.")
                st.stop()
            try:
                raw_text = uploaded.read().decode("utf-8")
                data = json.loads(raw_text)
            except Exception as e:
                st.error(f"Errore lettura JSON: {e}")
                st.stop()

            if not langs_list:
                st.warning("Inserisci almeno una lingua target.")
                st.stop()

            # Svuota i risultati precedenti PRIMA di tradurre
            st.session_state.translation_results.clear()

            base_name = uploaded.name
            name, ext = os.path.splitext(base_name)
            if not ext:
                ext = ".json"

            progress = st.progress(0)
            status = st.empty()

            total_langs = len(langs_list)

            for idx, lang in enumerate(langs_list, start=1):
                status.write(f"Traduzione in **{lang}** ({idx}/{total_langs})…")
                t0 = time.time()

                # Copia profonda per non accumulare traduzioni
                obj = json.loads(raw_text)

                def prog_cb(done_chunks: int, total_chunks: int):
                    # Barra cumulativa per questa lingua
                    base = (idx - 1) / total_langs
                    frac = done_chunks / max(1, total_chunks)
                    progress.progress(min(1.0, base + frac / total_langs))

                try:
                    translated = translate_all_human_text(obj, lang, api_key, progress_cb=prog_cb)
                except Exception as e:
                    st.error(f"Errore durante la traduzione {lang}: {e}")
                    continue

                dt = time.time() - t0
                log(f"{base_name} → {lang} completato in {dt:.2f}s")

                out_name = f"{name}.{lang}{ext}"
                out_bytes = json.dumps(translated, ensure_ascii=False, indent=2).encode("utf-8")
                st.session_state.translation_results[out_name] = out_bytes

            progress.progress(1.0)
            status.write("✅ Fatto.")

            if st.session_state.translation_results:
                st.success("Traduzione completata. Scarica i file qui sotto.")
            else:
                st.warning("Nessun file prodotto.")

            # Forza un piccolo rerun per mostrare i pulsanti anche se non c'è altro input
            _rerun()

    # --------- TAB: Log ---------
    with tabs[2]:
        st.subheader("Log")
        if not st.session_state.logs:
            st.write("Nessun log al momento.")
        else:
            st.code("\n".join(st.session_state.logs), language="text")


if __name__ == "__main__":
    main()
