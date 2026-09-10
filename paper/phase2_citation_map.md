# Phase-2 bibliography: what happens to each entry

Every numbered reference of the Phase-2 report, its fate in the paper, and the verified key in
`references.bib` that replaces it. "Delete" means the paragraph that cited it goes too, because
the paper's related work is thematic and those paragraphs were off topic.

| Phase-2 ref | Entry | Fate | Replacement key(s) |
|---|---|---|---|
| [1] | Calderon et al. 2023, NLG distillation | delete (off topic) | — |
| [2] | Hahn & Choi 2019, self-KD | delete (off topic) | — |
| [3] | "Anonymous" ACL submission | delete (not citable) | — |
| [4] | Zou et al., dynamic MT-KD for KBQA | keep, corrected: ESWA vol. 263, art. 125599, 2025, DOI 10.1016/j.eswa.2024.125599 | `zou2025dynamic` |
| [5] | Kaggle Penn Treebank mirror | delete (only supported [2]) | — |
| [6] | Merity et al., WikiText | delete (only supported [2]) | — |
| [7] | IBM blog on KD | delete; cite the survey and the original | `gou2021knowledge`, `hinton2015distilling` |
| [8] | Hinton et al. 2015 | keep (merge with [18]) | `hinton2015distilling` |
| [9] | Buciluă et al. 2006 | keep, with DOI | `bucilua2006model` |
| [10] | Schwartz et al., Green AI | keep (merge with [11]); cite CACM 2020, not arXiv; not for the GPT-3 claim | `schwartz2020green`, plus `brown2020language`, `patterson2021carbon` |
| [11] | duplicate of [10] | merge | — |
| [12] | Prasomphan, Thai MTKD | keep only with full venue, year, pages, DOI (verify_refs.py has no entry yet: find the conference record first) | to add |
| [13] | Gu et al., MiniLLM (cited as "Luo et al.") | delete the paragraph; cite MiniLLM only if LLM distillation is discussed, as ICLR 2024 | (not in list; add if needed) |
| [14] | Liu, Shen, Lapata 2020, noisy self-KD for summarisation | delete (off topic) | — |
| [15] | Wu, Wu, Huang 2021, MT-BERT | keep, cite Findings of ACL-IJCNLP 2021 with DOI | `wu2021one` |
| [16] | Yan et al. 2025, intrusion detection, low-tier venue | delete; adaptive KD is covered by peer-reviewed work | `liu2020adaptive`, `du2020agree`, `zhang2022confidence` |
| [17] | Shinde/Raut et al. 2025, low-tier venue | delete or move to a footnote; no peer-reviewed standing | — |
| [18] | duplicate of [8] | merge | — |
| [19]–[24] | BERTSUM, pointer-generator, Paulus, Gehrmann, BART, UniLMv2 | delete (supported [14]) | — |
| [25] | Furlanello et al., born-again networks | keep only if self-distillation is discussed; cite ICML 2018 | (add with verify_refs.py if used) |
| [26] | Moslemi et al., "ACM CSUR" with ScienceDirect URL | delete unless the real record is found; edge-KD motivation is covered by | `zhou2019edge`, `gou2021knowledge` |
| [27] | Belinga et al., venue/URL mismatch | delete (off topic, unverifiable) | — |
| [28] | Jobaer et al., venue/URL mismatch | delete (off topic, unverifiable) | — |
| [29] | UNICEF 2019 poll | keep with URL and access date | add as `@misc` |
| [30] | Vogels 2022, Pew | keep with URL and access date | add as `@misc` |
| [31] | Hamed et al. 2025, multimodal fake news | delete from KD claims | — |
| [32] | IBM edge-AI page | delete; cite the Proc. IEEE survey | `zhou2019edge` |
| [33] | Passalis et al. 2020 | keep, cited correctly as heterogeneous distillation | `passalis2020heterogeneous` |
| [34] | Qin et al. 2025, fake news | delete from deployment-gap claim | — |
| [35] | Sardar et al. 2025, IJRASET | delete; cyberbullying with DistilBERT is covered by peer-reviewed work | `rosa2019automatic`, `salawu2020approaches`, `emmery2021current` |
| [36] | Szczepanski et al. 2024 | delete from black-box/latency claim | — |

Missing from Phase 2 and now present in `references.bib`: the dataset papers
(`wulczyn2017exmachina`, `wang2020sosnet`), the models and tooling (`devlin2019bert`,
`liu2019roberta`, `sanh2019distilbert`, `turc2019wellread`, `he2023debertav3`,
`loshchilov2019decoupled`, `wolf2020transformers`, `paszke2019pytorch`), the cyberbullying and
implicit-abuse literature (section 2.1 of the related-work draft), the compact-model and
multi-teacher distillation literature (sections 2.2 and 2.3), and the annotator-disagreement
literature (section 2.4).
