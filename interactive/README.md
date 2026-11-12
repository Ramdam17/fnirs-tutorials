# Interactive illustrations

This folder holds standalone HTML and JavaScript illustrations embedded in book pages.

## Adding an illustration

An illustration is one HTML file in this folder, with its styles and its script written inside it. It may load one script, the vendored D3, by a relative address such as `<script src="vendor/d3.min.js"></script>`, and nothing else: no script from a content-delivery network, no web font, no image by address. The test `tests/test_interactive.py` fails when a file loads from another host, or loads a script that is not a relative address resolving to a file under `interactive/vendor/`, so that a page keeps working offline, on Colab and on the deployed site. An address that is only compared with a string, such as the SVG namespace, is not a load and is accepted. A link that a reader clicks (an `<a href>`) is not a load either; only the `href` of an anchor is left out of the scan, so a `ping` or an `onclick` in the same tag is still read.

The scan reads the text of the file, and it can only find an address that is written in it. It finds the usual forms of a load (a script, a stylesheet, an image, a frame, a form, a `<meta http-equiv="refresh">`, a `<base>`, a CSS `url()` or `@import`, `fetch`, `import`, a worker, a socket) with quoted or unquoted values, and any other address written after a quote, a backtick, a parenthesis or an equal sign. It cannot see an address assembled at run time, for example one built by joining pieces of text or read from data, so a page that builds an address that way is outside what the test guarantees. Review such a page by hand.

## Drawing approach

The illustrations are drawn with D3 7.9.2 (280,622 bytes, SHA-256 `912111098ea99503a6f4967361d2bbdc9dbf58cf9c678c8e27e5b02d1a07f0c1`), which is copied into this repository as `interactive/vendor/d3.min.js` with its `LICENSE` and loaded from there, so that there is no build step and no other host. To update the library, replace the file and its `LICENSE` with those of the npm package `d3` at the new version (`dist/d3.min.js`), then change the version, the size and the SHA-256 in `tests/test_interactive.py` (the table `VENDORED`) and in `COMPAT.md`, and in this paragraph; a test fails when the three disagree. The files of `interactive/vendor/` are an allow-list: a page may load only a script named in `VENDORED`, and a file in that folder that is neither named there nor a licence fails the suite, so a second library needs its own entry with its version, size and hash.

The folder is published as it is: the build copies everything under `interactive/` into the site, so a file at `interactive/banana.html` is served at `<base>/interactive/banana.html`, where the base is empty when the book is built locally and `/fnirs-tutorials` on the deployed book. A hidden file and a `__pycache__` folder are left out. This README is published too, at `<base>/interactive/README.md`.

## Embedding it in a page

A book page embeds an illustration with the `iframe` directive, and the source is the path from the root of the site, starting with `/interactive/`:

```
:::{iframe} /interactive/banana.html
:title: One sentence that describes the illustration for a screen reader
:width: 100%
:::
```

Write the source exactly like this and nothing else. A relative path, an address with `https://` and a path outside `/interactive/` are refused by the build. The base URL of the site is added in front of the source while the book is built, by the plugin `plugins/iframe-base-url.mjs`, so that the same page works at the root of a local server and under `/fnirs-tutorials` on the deployed book.

The frame takes its height from `book/interactive-frames.css`, which gives every illustration the same height. If an illustration needs more room, add a rule for its file next to the existing one, as the comment in that file explains, and check at a width of 390 px that the frame shows no scroll bar.

## What the build checks

`scripts/build_book.py` reads the pages it has built and refuses the book, with exit code 18, when a frame does not have the shape `<base>/interactive/<path>`, or when the file at that path is not in the built site, or when the HTML page that is deployed does not hold the same frames as the page file: every `<iframe>` of the HTML is read, whether or not the page file records it, and each must have the accepted shape. The file name in the source must also match the name of the file letter for letter, since the deployed host tells `Banana.html` from `banana.html` even where a Mac does not. The message names the page, the source it found and what is accepted. Run the notebook command first, then `uv run python scripts/build_book.py`; set `BASE_URL=/fnirs-tutorials` to build as the deployment does. A `BASE_URL` that is not empty and does not start with a single slash (for example `fnirs-tutorials`) is refused before the build, with exit code 19. A symbolic link in this folder that points out of it is refused too, with exit code 17, because the folder must publish only the files that lie in it.

## Previewing it

Open the HTML file directly in a browser to work on the drawing, since it needs no server: the file finds `vendor/d3.min.js` next to it. To see it in the book, build the site as above and serve the built folder with `python -m http.server --directory _build/executed/_build/html`; for a build with a base URL, serve a folder that holds the built site under a folder named `fnirs-tutorials`, so that the frame is requested at the address the page holds.
