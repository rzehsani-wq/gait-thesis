def parse_subject_from_folder(folder: str) -> int:
    # CASIA-B folder names often contain subject ID as the first 3 digits, e.g. "001-nm-01-000"
    # Fall back: parse first numeric group
    if folder is None:
        return -1
    s = str(folder)
    # common format: "001-..."
    try:
        head = s.split("-")[0]
        if head.isdigit():
            return int(head)
    except Exception:
        pass
    import re
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1

def parse_cond_from_folder(folder: str) -> str:
    # folder like "001-nm-01-000" → cond = "nm" (or "bg","cl")
    if folder is None:
        return "unk"
    s = str(folder).lower()
    parts = s.replace("\\", "/").split("-")
    if len(parts) >= 2:
        cond = parts[1][:2]     #nm / bg / cl
        if cond in {"nm", "bg", "cl"}:
            return cond
    return "unk"

def parse_view_from_folder(folder: str) -> int:
    """
    Parse view angle from CASIA-B folder name like '001-nm-01-090'.
    Format: <subject>-<condition>-<sequence>-<view>.
    parts[2] is sequence number (nm:01-06, bg/cl:01-02).
    part[3] is view angle (000-180).
    """
    if folder is None:
        return -1
    s = str(folder).rstrip("/\\")
    parts = s.split("-")
    if len(parts) >= 4:
        try:
            return int(parts[3])
        except ValueError:
            return -1
    return -1

def parse_seq_num_from_folder(folder: str) -> int:
    """
    Parse sequence number from CASIA-B folder name.
    nm: 01-06, bg: 01-02, cl: 01-02.
    """
    if folder is None:
        return -1
    s = str(folder).rstrip("/\\")
    parts = s.split("-")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except ValueError:
            return -1
    return -1