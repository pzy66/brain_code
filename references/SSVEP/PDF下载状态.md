# SSVEP PDF 下载状态

这个文件记录当前 `references/SSVEP` 里哪些论文 PDF 已经能随仓库保存，哪些只保留元数据和来源页面。

## 已保存 PDF

| 文献 | 本地 PDF | 来源说明 |
|---|---|---|
| Nakanishi et al., CCA methods comparison, 2015 | `01_core_method_papers/2015_nakanishi_cca_comparison_plosone.pdf` | PLOS open PDF, checked with `%PDF` header |
| Nakanishi et al., TRCA high-speed SSVEP speller, 2018 | `01_core_method_papers/2018_nakanishi_trca_pmc_author_manuscript.pdf` | PMC author manuscript PDF, checked with `%PDF` header and `pypdf` title/author/keyword extraction |
| Liu et al., TDCA, 2021 | `01_core_method_papers/2021_liu_tdca_tnsre_author_copy.pdf` | Author/open-source PDF, checked with `%PDF` header |
| Carrara and Papadopoulo, pseudo-online BCI evaluation, 2024 paper / 2023 preprint | `01_core_method_papers/2023_carrara_pseudo_online_arxiv.pdf` | arXiv PDF, checked with `%PDF` header |

## 已尝试但未保存 PDF

| 文献 | 当前保留材料 | 未保存 PDF 的原因 |
|---|---|---|
| Lin et al., CCA baseline, 2007 | Crossref JSON + PubMed page | IEEE endpoint did not return a valid shareable PDF |
| Bin et al., online multi-channel CCA, 2009 | Crossref JSON + Unpaywall JSON | IOP endpoint returned a Radware/HTML page, not a PDF |
| Chen et al., FBCCA, 2015 | Crossref JSON + PubMed page + Unpaywall JSON | IOP endpoint returned a Radware/HTML page, not a PDF |
| Wong et al., TRCA-R / spatial filtering, 2020 | Crossref JSON + PubMed page + Unpaywall JSON | IEEE endpoint returned an HTML/access-control page, not a PDF |
| Zhang et al., idle-state detection, 2015 | Crossref JSON + PubMed page + Unpaywall JSON | World Scientific endpoint returned an HTML page, not a PDF |
| Nakanishi et al., dynamic stopping, 2015 | Crossref JSON + PubMed page + Unpaywall JSON | IEEE conference endpoint did not return a valid shareable PDF |

## 维护原则

只把可以从开放、官方、作者或公共仓储来源取得，并且通过 `%PDF` 检查的文件放进 GitHub。ResearchGate、Sci-Hub、权限不清的转载文件不放进公开仓库。
