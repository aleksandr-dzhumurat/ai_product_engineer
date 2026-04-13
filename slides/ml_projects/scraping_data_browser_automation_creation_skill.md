# CloakBrowser Slot Monitoring Script — Development Guide

## Task

Build a Python polling script that monitors a booking website for available appointment slots and sends a Telegram notification when slots open up.

## Inputs always provided

| File | Purpose |
|---|---|
| `data/*.json` | Puppeteer recording — exact user actions, selectors, values |
| `data/*.js` | Same recording as a runnable JS file — use for selector reference |
| `data/*.mhtml` | Saved page snapshot — use to understand DOM structure, IDs, classes |

---

## Script structure

For implementation clicker script use reference nites below.

### Required features

1. **Polling loop** — random interval 5–10 min, weekdays only, within check window (default 09:00–15:30)
2. **Slot detection** — fill the form, trigger the check action, detect the "no slots" indicator
3. **Telegram notification** — only when slots ARE available (no TG on "no slots" to avoid flood)
4. **Browser event logging** — attach handlers after page creation:
   ```python
   page.on("console", lambda msg: logger.debug(f"[browser console] {msg.type}: {msg.text}"))
   page.on("pageerror", lambda err: logger.warning(f"[browser error] {err}"))
   page.on("framenavigated", lambda frame: logger.info(f"[browser nav] {frame.url}") if frame == page.main_frame else None)
   ```
5. **Per-field logging** — log each field name and value before filling

---

## Selector rules (critical — read carefully)

### Rule 1: Prefer stable ID selectors; fall back to get_by_label() for dynamic UUIDs
Use `page.locator("#fieldId")` when the ID is a stable short name — IDs are unique and bypass hidden-duplicate problems.
Use `page.get_by_label("Field Name")` **only** when the field's ID is a dynamic UUID (regenerated each session) **and** the field is outside any modal.

```python
# GOOD — stable short IDs (e.g. #label4, #birthDate, #label1001)
page.locator("#label4").fill(name)
page.locator("#label1001").fill(citizenship)

# GOOD — dynamic UUID id, field is outside modals → use get_by_label()
page.get_by_label("Re-enter the email address").fill(email)

# BAD — get_by_label() on a field inside or near a modal: matches hidden duplicates still in DOM
page.get_by_label("Citizenship").fill(citizenship)
```

### Rule 2: Dynamic UUID IDs — check if they're static or session-generated

Some UUID-looking IDs are **static** (same across sessions), others are **dynamic** (regenerated per session).

**How to tell:** Cross-reference the ID in the `.mhtml` snapshot with a live session run. If the element isn't found, the ID is dynamic.

For **static** UUID IDs (confirmed in `.mhtml` and stable in recording), use XPath since CSS cannot start with a digit:
```python
# GOOD — static UUID ID
page.locator('xpath=//*[@id="22c5017f-589b-4e30-8347-cc2226fb4572"]').fill(value)

# BAD — CSS escaping is fragile and breaks silently
page.locator("#\\33 3213c85-2639-41f9-b909-179175bb7bb0").fill(value)
```

For **dynamic** UUID IDs (change per session — XPath will fail with "element not found in DOM"), use `aria/` selector from the `.js` recording or `get_by_label()` **only if the field is outside modals** (no hidden duplicates):
```python
# GOOD — for a field OUTSIDE modals with a dynamic UUID id
page.get_by_label("Re-enter the email address").fill(value)

# BAD — dynamic UUID, not reliable across sessions
page.locator('xpath=//*[@id="33213c85-2639-41f9-b909-179175bb7bb0"]').fill(value)
```

**How to find the aria label name:** check the `.js` recording file — it lists selectors in priority order, with `aria/Field Name` first.

> **Rule of thumb:** If XPath by ID throws "element not found in DOM" (not timeout, not invisible — specifically "not found"), the ID is dynamic. Switch to the aria/label approach.

### Rule 3: Modal interactions — scope to the modal element
Items inside modals start hidden. The search input unhides matching items.
Always scope locators to the modal, not the full page.

```python
modal = page.locator("#modalId")
modal.wait_for(state="visible", timeout=8000)
search = modal.locator("input[type='text']")
search.click()
search.fill("search term")
time.sleep(1)
label = modal.locator("label[for='item-uuid']")
label.wait_for(state="visible", timeout=5000)
label.click()  # clicking label activates radio/checkbox AND closes modal
# DO NOT click the radio/checkbox again after this — it's detached from DOM
```

### Rule 4: After clicking a label that closes a modal — do NOT click the underlying input
The modal closes on label click, detaching the input from the DOM. A second click will fail with "element not found in DOM".

### Rule 5: Verify selectors against the MHTML
Before writing a selector, confirm the element structure in the `.mhtml` file.
- Check class names, IDs, nesting
- Check if items have `hidden` class by default (means search filtering is required)
- Check if the element is inside a modal (`display: none` initially)

---

## Slot detection pattern

```python
# Click the action button that reveals slot availability
page.locator("selector for check button").click()
time.sleep(3)

# If "no slots" modal appears → fully booked
try:
    no_slots_indicator = page.locator("#modal-id button")  # use ID from recording/mhtml
    no_slots_indicator.wait_for(state="visible", timeout=4000)
    logger.info("No slots — fully booked.")
    no_slots_indicator.click()  # dismiss
    return False
except TimeoutError:
    # No blocking modal → slots available
    logger.info("SLOTS AVAILABLE!")
    return True
```

---

## Telegram rules

```python
def notify_telegram(text: str) -> None:
    if TG_BOT_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text})

# In the loop:
if slots_available:
    notify_telegram(f"@user SLOTS AVAILABLE! Book now!\n{BOOKING_URL}")
else:
    logger.info("No slots. Next check in X min...")  # NO TG here
```

To disable TG during development:
```python
def notify_telegram(text: str) -> None:
    logger.info(f"[TG DISABLED] {text}")
    # comment out actual sending
```

---

## Extracting screenshots from a session recording video

When debugging using a screen recording alongside a log file, extract frames at the timestamps logged.

```bash
# Extract a frame at a given offset from the video start
ffmpeg -ss 00:01:23 -i recording.mp4 -frames:v 1 screenshot.png
```

**Timestamp offset is approximate.** The video may start a few seconds before or after the script begins, so log timestamps and video timestamps will not align exactly. If the extracted frame looks wrong, shift the offset by ±5–10 seconds and re-extract.

Workflow:
1. Find the log line of interest (e.g. `2026-06-03 10:15:42 [INFO] Clicking Book button.`)
2. Note the wall-clock time from the log.
3. Subtract the video start time (check video file metadata or the first visible action) to get the approximate offset.
4. Extract the frame with `ffmpeg -ss <offset>` and inspect it.
5. If the frame is off, adjust the offset iteratively.

---

## Common mistakes to avoid

| Mistake | Fix |
|---|---|
| `get_by_label()` on a page with modals | Use `#id` selector instead |
| `page.locator("#\33 uuid...")` for UUID IDs | Use `xpath=//*[@id="uuid..."]` for static UUIDs |
| XPath by UUID fails with "element not found in DOM" (not timeout) | UUID is dynamic — use `page.get_by_label("Field Name")` for fields outside modals; check `.js` recording for `aria/Field Name` selector |
| Clicking radio/checkbox after clicking its label | Label click is sufficient; element detaches after modal closes |
| `text/Some text` locator for items inside search-filtered modals | Scope to modal + use `label[for='...']` after triggering search |
| Sending TG on every "no slots" check | Only send TG when slots ARE found |
| Breaking out of polling loop on slot found | Keep monitoring — just notify TG and continue |
| `input()` / blocking calls in the loop | Remove all blocking interactive calls from the loop |
