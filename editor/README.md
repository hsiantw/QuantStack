# QuantStack rich-text editor

The notebook uses locally bundled Tiptap/ProseMirror and DOMPurify. No editor
service, CDN, API key, or network request is needed while writing notes.
The Python server serves the checked-in `web/note-editor.js` artifact.

To rebuild after changing `note-editor.js`, use Node 24 LTS and run in this folder:

```
npm ci --ignore-scripts
npm run build
```

Commit the source, lockfile, generated bundle and generated third-party licenses.
All direct dependencies are pinned. Build tools are not required on Render.

`body` remains the plain-text search/preview field for legacy backup compatibility.
Optional `bodyHtml` stores formatting (maximum 200,000 characters), with a 20,000
character plain-text limit. Old notes are escaped as plain text. On load, paste
and export, HTML passes through a strict DOMPurify allowlist and the editor schema.
Only HTTP(S) and mailto links are allowed; scripts, embeds, images, remote styles,
event handlers and arbitrary CSS are removed. Table spans and widths are bounded.
Undo history resets when opening another note. Exports include only the current
note; full notebook/checklist/category backups remain in the existing JSON flow.

Upstream references:
- https://tiptap.dev/docs/editor/getting-started/install/vanilla-javascript
- https://tiptap.dev/docs/editor/extensions/functionality/starterkit
- https://tiptap.dev/docs/editor/extensions/nodes/table
- https://github.com/cure53/DOMPurify
