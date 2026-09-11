"""Every \\cite key in the drafts must exist in references.bib, and every bib entry should be used."""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "paper")
bib = set(re.findall(r"@\w+\{([^,]+),", (root / "references.bib").read_text(encoding="utf-8")))
used, where = set(), {}
for p in sorted(root.glob("*.md")):
    text = p.read_text(encoding="utf-8")
    for group in re.findall(r"\\cite\{([^}]*)\}", text):
        for k in (x.strip() for x in group.split(",")):
            if k:
                used.add(k)
                where.setdefault(k, set()).add(p.name)

missing = sorted(used - bib)
unused = sorted(bib - used)
print(f"{len(bib)} entries in references.bib, {len(used)} distinct keys cited")
if missing:
    print("\nCITED BUT NOT IN THE BIBLIOGRAPHY (these would break the build):")
    for k in missing:
        print(f"  {k}  <- {', '.join(sorted(where[k]))}")
else:
    print("every cited key resolves")
if unused:
    print(f"\nin the bibliography but not yet cited ({len(unused)}), fine while drafting:")
    print("  " + ", ".join(unused))
sys.exit(1 if missing else 0)
