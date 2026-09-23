# BadWAM Project Page

This folder is a self-contained static project page for:

**BadWAM: When World-Action Models Dream Right but Act Wrong**

## Local preview

```bash
cd BadWAM-main/docs
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## GitHub Pages deployment

You can push the contents of this folder to a GitHub repository and enable
GitHub Pages from the repository settings.

Recommended layout:

```text
your-repo/
├── index.html
├── style.css
├── script.js
├── .nojekyll
└── assets/
    ├── images/
    └── paper/
```

The page has no external JavaScript or CSS dependencies.
