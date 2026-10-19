# Shot list: 45 photos

**Before every photo: no personal details.** No names (prescriptions, pharmacy stickers with a patient name),
no addresses, no phone numbers, no e-mail addresses, no faces, no card or account numbers, no receipts with card
digits. Cover them with a finger or a sticky note, or choose another item. Photos stay in the git-ignored
`eval/data/private/`; book pages are copyrighted, so they are never committed.

How (easiest): put all the photos in **`eval/data/private/inbox/`** (git-ignored). Subfolders named after a category
(`inbox/medicine_strip/`, `inbox/book_page/`, ...) fill in the category; this is optional. Then:

```bash
.venv/bin/python eval/tools/import_folder.py      # copies them in: rotation fixed, all metadata (GPS) removed
.venv/bin/python eval/tools/collect.py            # open http://localhost:7871 and set the details on each photo
```

Photos still missing a detail are listed first, outlined in red. Your originals in `inbox/` are not changed; delete
them yourself once imported if you like (they still contain the GPS location). The page can also take webcam photos
(**Take photo**, captured like the app's Capture now) or **Add files** directly.

One item can appear in at most 2 photos, and then under different conditions (for example good light and glare).

Hold the phone or item the way a low-vision user might: roughly centred, but not perfectly framed.

| # | Category | Shots | Conditions to cover (lighting / angle) |
|---|---|---|---|
| 1 | `medicine_strip` (foil blister strip, back side with batch / mfg / exp) | 8 | 3 good/flat · 2 **glare** (shiny foil under a lamp) · 1 **dim** · 1 tilted · 1 curved (strip bent) |
| 2 | `medicine_box` (carton or bottle label: name, dosage, warnings) | 8 | 3 good/flat · 1 **glare** · 1 **dim** · 2 tilted · 1 curved (bottle label) |
| 3 | `food_label` (ingredients, best before / use by, MRP or price) | 8 | 3 good/flat · 1 **glare** (plastic wrap) · 1 **dim** · 1 tilted · 2 curved (can, packet) |
| 4 | `book_page` (single pages and two-page spreads with page numbers) | 8 | 3 good/flat (2 of them spreads) · 1 **dim** · 1 **glare** (glossy page) · 1 tilted · 2 curved (open book near the spine) |
| 5 | `document` (manual, leaflet, printed notice, instructions; nothing personal) | 7 | 3 good/flat · 1 **dim** · 1 **glare** · 2 tilted |
| 6 | `sign_or_screen` (printed sign, laptop or phone screen, appliance display) | 6 | 3 good/flat · 1 **glare** (screen reflection) · 1 tilted · 1 dim |
| | **Total** | **45** | **16 deliberately difficult** (dim or glare), plus 8 tilted and 9 curved |

Also include:

- **Small print:** at least 5 photos where the key text is small (medicine strip batch / exp lines, food ingredients).
  Write "small print" in Notes.
- **Expired items:** 2–3 items whose expiry date has already passed (before 19 Oct 2026), so "is it expired?" has
  both answers. Write "expired" in Notes.
- **Expiry wording variety:** where you have them, items printed as `Exp. Date 02/2027`, `EXP 04/28`, `Use by`,
  `Best before`.
- **Devices:** about 35 phone and 10 laptop webcam photos, spread across the categories.
- **Dim** means a lamp off or the curtains closed, still readable to you. **Glare** means a bright reflection over
  part of the text.
