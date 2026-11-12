// MyST plugin: put the base URL of the site in front of the source of an interactive frame.
//
// What it does. A page of the book embeds an illustration with
//
//     :::{iframe} /interactive/<file>.html
//     :::
//
// The source is a path from the root of the site, and MyST keeps it exactly as written: it
// does not add the base URL of the build. On the deployed book the base URL is
// /fnirs-tutorials, so the browser would ask for /interactive/<file>.html and get a 404.
// This transform rewrites a source that starts with /interactive/ to
// <BASE_URL>/interactive/<file>.html. Every other source is left untouched, so that the
// check of scripts/build_book.py sees what the author wrote. With no BASE_URL it changes
// nothing. It prints one line for each frame it rewrites, with the source before and after.
//
// Why a plugin. It works on the tree MyST parsed, so it also sees a frame that an
// {include} brings in, and it works when an author runs `jupyter book start` or
// `jupyter book build` without scripts/build_book.py.
//
// Where the base URL comes from. The plugin reads process.env.BASE_URL, the variable that
// MyST itself reads to build the site (https://mystmd.org/guide/deployment). The plugin
// documentation does not promise that a plugin sees it
// (https://mystmd.org/guide/plugins-reference gives a transform no configuration), so the
// result is verified rather than trusted: check_frames in scripts/build_book.py reads the
// built page and refuses a frame whose source is not <BASE_URL>/interactive/<file> with
// the file published in the built site.
//
// Tests. The plugin has no unit test. It runs inside the build, so it is exercised by the
// real builds only: the book build with no base URL, the book build with BASE_URL set, and
// the build on the CI runner, where BASE_URL is set. check_frames is the test.

const INTERACTIVE_PREFIX = '/interactive/';

/** Return the base URL without its trailing slash, or '' when none is set. */
function baseUrl() {
  return (process.env.BASE_URL || '').replace(/\/+$/, '');
}

const iframeBaseUrl = {
  name: 'iframe-base-url',
  doc: 'Prefix the source of an iframe that starts with /interactive/ with the BASE_URL of the build.',
  stage: 'document',
  plugin: (_options, utils) => (tree) => {
    const base = baseUrl();
    if (!base) return;
    for (const node of utils.selectAll('iframe', tree)) {
      if (typeof node.src !== 'string' || !node.src.startsWith(INTERACTIVE_PREFIX)) continue;
      const before = node.src;
      node.src = base + before;
      console.log(`iframe-base-url: ${before} -> ${node.src}`);
    }
  },
};

export default {
  name: 'Base URL for interactive frames',
  transforms: [iframeBaseUrl],
};
