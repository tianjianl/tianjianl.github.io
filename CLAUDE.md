# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Tianjian Li's personal academic website (`tianjianl.github.io`), built on the
[al-folio](https://github.com/alshedivat/al-folio) Jekyll theme. Most of the repo is upstream
theme machinery; day-to-day edits touch a small set of content files (see below). Deployed to
GitHub Pages.

## Commands

```bash
bundle install                              # install Ruby gem dependencies (first time)
bundle exec jekyll serve                    # local dev server with live reload at localhost:4000
bundle exec jekyll build --lsi              # production build (--lsi = latent semantic indexing for related posts); same as bin/cibuild
./bin/deploy                                # build + force-push _site to gh-pages branch (prompts for confirmation)
docker compose -f docker-local.yml up       # build & run in Docker from local Dockerfile
docker compose up                           # run via prebuilt dockerhub image (port 8080)
```

Pre-commit hooks (`.pre-commit-config.yaml`) enforce trailing-whitespace / EOF / YAML checks —
run `pre-commit install` once, or `pre-commit run --all-files` manually.

## Where content lives (edit these)

- `_pages/about.md` — homepage bio. Front matter toggles `news`, `selected_papers`, `social`, etc.
- `_bibliography/papers.bib` — **all publications**. BibTeX entries; custom fields drive rendering:
  `selected={true}` surfaces a paper on the homepage; `arxiv=`, `url=`, `pdf=`, `code=`, `abstract=`
  produce the link buttons. Edited far more often than anything else.
- `_news/*.md` — news/announcement items shown on the homepage. One file per item, dated via front matter.
- `_data/cv.yml` + `_pages/cv.md` — CV content. `_data/coauthors.yml` bolds/links coauthor names in the
  publication list. `_data/venues.yml` maps venue abbreviations to full names/colors.
- `assets/` — images (`assets/img/`), PDFs, and the profile picture (`assets/img/prof_pic.jpg`).
- `_config.yml` — site-wide settings (title, social handles, theme, plugin config). A Jekyll restart is
  required after changing it (live reload does not pick up `_config.yml` edits).

## Architecture notes

- **Jekyll static site.** Pages live in `_pages/`, layouts in `_layouts/`, reusable partials in
  `_includes/`, styles in `_sass/`. `_site/` is the generated output — never edit it (gitignored).
- **Publications are generated from BibTeX**, not hand-written HTML, via the `jekyll-scholar` plugin.
  The bibliography rendering is configured under the `scholar:` key in `_config.yml`.
- **Custom plugins** in `_plugins/`: `details.rb` (collapsible blocks), `hideCustomBibtex.rb` (strips
  the custom `.bib` fields above out of the displayed BibTeX), `external-posts.rb` (pulls in external blog posts).
- **Deployment**: `.github/workflows/deploy.yml` builds and deploys on push to `master` (the source
  branch); the rendered site is served from `gh-pages`. `bin/deploy` does the same manually. Do not commit
  build artifacts to `master`.

## Conventions

- Adding a publication = add a BibTeX entry to `papers.bib`; do not edit publication HTML directly.
- Keep new coauthors in `_data/coauthors.yml` so their names render consistently across papers.
