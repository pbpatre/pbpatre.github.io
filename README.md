# Pratik Patre — Technical Blog

> **Live site:** https://pbpatre.github.io/ (custom domain: `pratikpatre.dev` — configure in `static/CNAME`)

A professional technical blog for a Principal ML Systems Engineer at Atlassian, built with [Hugo](https://gohugo.io/) + [PaperMod](https://github.com/adityatelange/hugo-PaperMod) and deployed to GitHub Pages via GitHub Actions.

---

## Quick Start

### Prerequisites

- [Hugo Extended](https://gohugo.io/installation/) v0.124.0+
- Git

### Local Development

```bash
# 1. Clone the repository (includes PaperMod submodule)
git clone --recurse-submodules https://github.com/pbpatre/pbpatre.github.io.git
cd pbpatre.github.io

# 2. Add PaperMod theme (first time only)
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive

# 3. Start local dev server
hugo server -D --bind 0.0.0.0

# Site available at http://localhost:1313
```

### Create a New Post

```bash
# New post in a section (choose: llm-production, ml-engineering, research, engineering)
hugo new llm-production/my-new-post.md

# Edit the generated file in content/llm-production/my-new-post.md
# Set draft: false when ready to publish
```

### Build for Production

```bash
hugo --gc --minify
# Output in ./public/
```

---

## Repository Structure

```
pbpatre.github.io/
├── .github/workflows/deploy.yml   # GitHub Actions CI/CD
├── config.yml                     # Hugo configuration
├── archetypes/default.md          # New post template
├── assets/css/extended/custom.css # Custom styling
├── content/
│   ├── _index.md                  # Homepage
│   ├── about.md                   # About page
│   ├── search.md                  # Search page
│   ├── llm-production/            # LLM Production posts
│   ├── ml-engineering/            # ML Engineering posts
│   ├── research/                  # Research & paper reviews
│   └── engineering/               # Engineering & infrastructure
├── layouts/
│   ├── partials/
│   │   ├── extend_head.html       # KaTeX, Mermaid, GA4, OG tags
│   │   └── extend_footer.html     # Custom footer
│   └── shortcodes/
│       ├── callout.html           # Callout boxes (note/tip/warning/danger/info)
│       └── series-nav.html        # Multi-part series navigation
├── static/
│   ├── images/                    # profile.jpg, og-image.jpg, favicon.ico
│   ├── robots.txt
│   └── CNAME                      # Custom domain (add pratikpatre.dev)
└── themes/PaperMod/               # Git submodule
```

---

## Configuration

### 1. Google Analytics

Replace the placeholder in `config.yml`:
```yaml
googleAnalytics: "G-XXXXXXXXXX"  # → your real GA4 measurement ID
```

### 2. Custom Domain

```bash
echo "pratikpatre.dev" > static/CNAME
```
Then configure the DNS A records for GitHub Pages in your domain registrar.

### 3. Social Links

Update in `config.yml` under `params.socialIcons`:
```yaml
- name: email
  url: "mailto:your.real.email@example.com"
```

### 4. Profile Image

Replace `static/images/profile.jpg` with a 400×400px optimised JPEG.  
Replace `static/images/og-image.jpg` with a 1200×630px image for social sharing.

---

## Writing Guide

### Front Matter Reference

Every post should include:

```yaml
---
title: "Your Post Title"
date: 2024-06-01
draft: false
author: "Pratik Patre"
description: "One-sentence SEO description (150-160 chars)"
summary: "2-3 sentence summary shown in post listings"
tags: ["Tag1", "Tag2", "Tag3"]
categories: ["LLM Production"]   # matches your section
series: ["Series Name"]          # optional, for multi-part posts
series_weight: 1                 # order within series
cover:
  image: "images/posts/my-cover.jpg"
  alt: "Description of the image"
ShowToc: true
TocOpen: true
---
```

### Shortcodes

**Callout boxes:**
```
{{</* callout type="note" title="Custom Title" */>}}
Your content here. Supports **markdown**.
{{</* /callout */>}}
```
Types: `note` | `tip` | `warning` | `danger` | `info`

**Series navigation:**
```
{{</* series-nav series="LLM Production Systems" */>}}
```

**Math (KaTeX):**
```
Inline: $E = mc^2$
Display: $$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
```

**Mermaid diagrams:**
````
```mermaid
graph LR
    A[Client] --> B[API Gateway]
    B --> C[Model Server]
```
````

---

## Deployment

Deployment is fully automated via GitHub Actions (`.github/workflows/deploy.yml`).

**Trigger:** Every push to `main` builds and deploys the site.

### GitHub Pages Setup (one-time)

1. Go to **Settings → Pages** in your GitHub repository
2. Set **Source** to **GitHub Actions**
3. Push to `main` — the workflow handles the rest

### Manual Deploy

```bash
# Trigger a deploy without a code change
gh workflow run deploy.yml
```

---

## Performance

- **Lighthouse target:** > 95 across all categories
- **Page weight:** < 500KB (excluding images)
- **Build time:** < 10 seconds

Key optimisations:
- `minify: true` in Hugo config
- KaTeX loaded deferred, only renders on pages with math
- Mermaid loaded as ES module (no blocking render)
- PaperMod uses minimal JavaScript by default

---

## Content Sections

| Section | URL | Focus |
|---------|-----|-------|
| LLM Production | `/llm-production/` | Deployment, scaling, cost, monitoring |
| ML Engineering | `/ml-engineering/` | Feature stores, serving, MLOps |
| Research | `/research/` | Paper reviews, experiments |
| Engineering | `/engineering/` | Kubernetes, infra, system design |

---

## License

Content © Pratik Patre. All rights reserved.  
Theme (PaperMod) licensed under MIT.
