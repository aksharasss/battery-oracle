---
name: Battery Oracle 2.0
description: Second-life cell grading written on one sheet of engineering graph paper, graphite ink and a single red pen.
colors:
  dark-ground: "#0d1611"
  dark-panel: "#101b14"
  dark-grid: "#1b2c22"
  dark-grid-major: "#2d4a37"
  dark-select: "#29473a"
  dark-rule: "#55705f"
  dark-fresh: "#8fae9b"
  dark-muted: "#a3bcab"
  dark-ink: "#eaf2ec"
  dark-pen: "#ff7a66"
  light-ground: "#f6faf7"
  light-panel: "#eef5f0"
  light-grid: "#d6e8dc"
  light-grid-major: "#afd0ba"
  light-select: "#cfe5d6"
  light-rule: "#1d2227"
  light-fresh: "#5d6b65"
  light-muted: "#4b5a54"
  light-ink: "#1d2227"
  light-pen: "#c42a22"
typography:
  display:
    fontFamily: "Archivo, Segoe UI, sans-serif"
    fontSize: "8.5rem"
    fontWeight: 900
    lineHeight: 0.82
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "1.9rem"
    fontWeight: 700
    lineHeight: 1.15
  statement:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.3
  title:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 700
  mast:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "1.35rem"
    fontWeight: 700
  body:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
  body-sm:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "0.95rem"
    fontWeight: 400
    lineHeight: 1.45
  data:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 700
    fontFeature: "\"tnum\""
  label:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 400
    letterSpacing: "0.08em"
  stamp:
    fontFamily: "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 700
    letterSpacing: "0.1em"
  hand:
    fontFamily: "Caveat, Segoe Print, cursive"
    fontSize: "24px"
    fontWeight: 600
rounded:
  none: "0px"
spacing:
  xs: "0.35rem"
  sm: "0.6rem"
  md: "1rem"
  lg: "1.4rem"
  xl: "2rem"
  grid-minor: "10px"
  grid-major: "50px"
components:
  grade-letter:
    textColor: "{colors.dark-ink}"
    typography: "{typography.display}"
  grade-letter-c:
    textColor: "{colors.dark-pen}"
    typography: "{typography.display}"
  stamp:
    textColor: "{colors.dark-pen}"
    typography: "{typography.stamp}"
    rounded: "{rounded.none}"
    padding: "0.3rem 0.6rem"
  note:
    textColor: "{colors.dark-ink}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.none}"
    padding: "0.6rem 0.85rem"
  note-tag:
    textColor: "{colors.dark-muted}"
    rounded: "{rounded.none}"
    padding: "0 0.35rem"
  note-tag-pen:
    textColor: "{colors.dark-pen}"
    rounded: "{rounded.none}"
    padding: "0 0.35rem"
  mechanism-bar-fill:
    backgroundColor: "{colors.dark-ink}"
    height: "1.25rem"
  mechanism-bar-fill-dominant:
    backgroundColor: "{colors.dark-pen}"
    height: "1.25rem"
  button:
    textColor: "{colors.dark-ink}"
    rounded: "{rounded.none}"
  button-hover:
    textColor: "{colors.dark-pen}"
    rounded: "{rounded.none}"
  select:
    backgroundColor: "{colors.dark-ground}"
    textColor: "{colors.dark-ink}"
    rounded: "{rounded.none}"
  menu-option-selected:
    backgroundColor: "{colors.dark-select}"
    textColor: "{colors.dark-ink}"
  sidebar:
    backgroundColor: "{colors.dark-panel}"
    textColor: "{colors.dark-ink}"
  chart:
    backgroundColor: "{colors.dark-ground}"
    height: "340px"
---

# Design System: Battery Oracle 2.0

## Overview

**Creative North Star: "The Lab Notebook Sheet"**

The whole result is one page of engineering graph paper. A millimetre grid with a heavier line every fifth millimetre covers the app background. The grade, the mechanism split and the dQ/dV curve are written on it in graphite ink, and anything that changed or needs a warning is marked with one red pen. It is not a dashboard of cards. There are no tiles and no shadows, and the only filled shapes are the ink bars. Everything is set apart with one-pixel rules, bordered stamps and the grid itself.

There are two sheets of the same notebook, and dark comes first. The dark sheet is a green-black ground with near-white graphite. The light sheet is a pale green-white ground with dark graphite. Both use the same red pen, retuned for contrast on each ground. Every color token comes in a dark/light pair with the same role. Switching sheets swaps values, never structure.

Hierarchy comes from scale contrast, not decoration. One huge display letter for the grade sits over a single legible UI face at two weights, and a handwritten face appears only where the pen annotates the chart. States and caveats are stamps that stay on the record. They are never toasts, alert boxes or tinted banners.

**Key Characteristics:**
- Millimetre graph-paper ground (10px minor, 50px major) behind everything, visible through every transparent container.
- Three inks per sheet: ground, graphite, red pen. Muted graphite is used for secondary text and rules.
- Square corners everywhere (0px). Rules are 1px. Heavier 2px borders are reserved for pen stamps.
- No shadows, no gradients except the grid itself, no card fills.
- The grade letter is the only display-face text. The chart annotation is the only handwriting.

## Colors

Each sheet has a green-tinted graphite family plus one warm red pen, and the pen is the only accent.

### Primary
- **Red Pen** (`dark-pen` on the dark sheet, `light-pen` on the light sheet): Marks what changed or what warns. Used for the dominant mechanism's bar fill and underline, a grade-C letter, the method stamp, borders and tags of warning notes, the chart's handwritten note and arrow, the version number in the masthead, focus rings, text selection, the spinner arc and hover on outlined controls.

### Neutral
- **Notebook Ground** (`dark-ground` / `light-ground`): The sheet itself. App background, chart paper and plot background, code blocks, select fields.
- **Margin Column** (`dark-panel` / `light-panel`): A one-step tint of the ground for the sidebar, dropdown menus and chart hover labels. It is the only tinted surface.
- **Minor Grid** (`dark-grid` / `light-grid`): The 10px millimetre lines and the chart's minor gridlines.
- **Major Grid** (`dark-grid-major` / `light-grid-major`): The 50px lines, chart major gridlines and table row dividers.
- **Selection Wash** (`dark-select` / `light-select`): Hovered or selected options in dropdown menus only.
- **Rule** (`dark-rule` / `light-rule`): Every one-pixel structural line. Masthead and header underlines, the column divider, control and note borders, bar tracks and their 10% ticks, chart axes, the scrollbar thumb. On the light sheet it is the same value as ink.
- **Fresh Reference Graphite** (`dark-fresh` / `light-fresh`): The dashed fresh-cell reference curve on the dQ/dV chart, and nothing else.
- **Muted Graphite** (`dark-muted` / `light-muted`): Secondary text. Subtitles, lead lines, fact labels, widget labels, neutral note tags, table headers, axis ticks and titles, captions.
- **Graphite Ink** (`dark-ink` / `light-ink`): Primary text, the grade letter for A and B, non-dominant mechanism bar fills, and the measured cell's solid curve.

### Named Rules
**The Three Inks Rule.** A sheet carries ground, graphite and red pen. Muted graphite, rules and grid lines are lighter strokes of the same graphite. No new hue enters a sheet.

**The One Red Pen Rule.** The pen only marks change, warning or annotation. Its only filled areas are the dominant mechanism bar and the grade-C letter. It is never a decorative fill or background.

**The Paired Sheets Rule.** Every color has a dark and a light value with the same role. A new color is not done until both values exist and both pass contrast on their own ground.

## Typography

**Display Font:** Archivo 900 (with Segoe UI, sans-serif)
**Body Font:** Atkinson Hyperlegible 400/700 (with Segoe UI, system-ui, sans-serif)
**Hand Font:** Caveat 600 (with Segoe Print, cursive)

**Character:** Atkinson Hyperlegible is built for legibility at a distance, and its slashed zero and open forms suit a projector. Archivo Black-weight gives the grade letter the mass of a rubber stamp. Caveat is the student's pen in the margin.

### Hierarchy
- **Display** (Archivo 900, 8.5rem, line-height 0.82, -0.02em; 6rem under 760px): The grade letter only. It is given an `aria-label` and never relies on color for meaning.
- **Headline** (700, 1.9rem, 1.15): The cell or file name at the top of the result sheet.
- **Statement** (700, 1.25rem, 1.3): The one-line second-life use under the grade.
- **Mast** (700, 1.35rem): The "Battery Oracle 2.0" masthead in the sidebar, with the version number in pen.
- **Title** (700, 1.15rem): Section heads on the sheet: mechanism breakdown and the dQ/dV curve.
- **Body** (400, 1rem): Facts, mechanism names (1.05rem), empty-state copy (1.1rem).
- **Body small** (400, 0.95rem, 1.45): Lead lines, note text and the peak table. Captions are 0.92rem, capped at 75ch.
- **Data** (700, tabular figures): Fact values, percentages and table numerals all use `font-variant-numeric: tabular-nums`.
- **Label** (400, 0.78rem, 0.08em, uppercase, muted): Form-widget labels in the sidebar.
- **Stamp** (700, 0.78rem, 0.1em, uppercase): The method stamp. Note tags use the same treatment at 0.72rem.
- **Hand** (Caveat 600, 24px, pen): The handwritten chart note only.

### Named Rules
**The Three Faces Rule.** Atkinson Hyperlegible sets every piece of UI text. Archivo is only for the grade letter, and Caveat is only for the red-pen note on the chart. Neither may spread to headings, buttons or labels.

**The Scale Contrast Rule.** Rank comes from size, using two weights (400, 700) of one face. Don't use color, boxes or extra weights to rank text.

## Layout

A single result sheet sits to the right of a sidebar. The sidebar is the notebook's margin column, holding the masthead, upload, sample picker, sheet toggle and file-format expander. The main column is capped at 1240px (top padding 1.2rem, bottom 3rem).

The sheet reads top to bottom. A header row puts the cell name and subtitle on the left and the method stamp on the right, closed by a one-pixel rule. Below it are two columns at 5:7 with a large gap. The left column has the grade letter, statement, fact list (an auto/1fr definition grid with 0.35rem by 1.2rem gaps) and stacked notes. The right column has the mechanism bars, separated by a 1px left rule with 2rem of inset, and then the dQ/dV chart (340px tall, indented 2rem, ruled top and bottom). Peak features sit in a full-width expander underneath.

Spacing follows the rem steps in the frontmatter. Section gaps are 1.4rem, and bars sit 1.05rem apart. Under 760px the columns stack. The left rule becomes a top rule, the indents drop to zero and the grade letter shrinks to 6rem.

**The Grid Shows Through Rule.** Containers on the sheet are transparent so the millimetre grid stays continuous behind notes, bars and tables. Only the sidebar, menus and the chart's own paper are filled.

## Elevation & Depth

The system is completely flat, with no shadows anywhere. Depth comes from two things. The sidebar sits on a one-step tint (the margin column) behind a 1px right rule. Stamps get emphasis from heavier borders (2px pen) and a -3° rotation, not from lift.

**The Flat Paper Rule.** Nothing floats above the sheet. Don't add `box-shadow`, blur or layered surfaces. If something needs emphasis, give it a heavier rule or the pen.

## Shapes

Every corner is square (0px). That includes the uploader dropzone, buttons, selects, expanders, code blocks, notes, stamps and bar tracks. Lines come in three weights. 1px is the structural rule, 1.5px is a neutral note border and 2px is the pen (method stamp, warning notes). The dropzone and the empty state use a 1px dashed rule to mark a place where something will be written. Bar tracks carry 1px ticks every 10% of the shared scale, like a ruler.

## Components

### Grade Letter (signature)
The grade is printed huge in Archivo 900. A and B are in graphite ink, and C is in red pen. The letter itself carries the grade, and the color only reinforces it.

### Method Stamp (signature)
A rubber stamp: inline-block, 2px pen border, pen text, the stamp type style, 0.3rem by 0.6rem padding, rotated -3°, no wrap. It names the grading path (slow twin fit, fast 4C model, placeholder) at the right of the header.

### Notes
Caveats are stamped onto the sheet and stay there.
- **Neutral note:** 1.5px rule border, transparent ground, 0.6rem by 0.85rem padding, body-small text. It opens with a tag: uppercase 0.72rem, 0.1em tracking, 1px currentColor box, muted.
- **Pen note:** 2px pen border with a pen-colored tag. Used for out-of-range, outside-calibration, error and placeholder-grader notices.

### Mechanism Bars (signature)
A name/value row (the value in bold tabular figures) above a 1.25rem track with a 1px rule border and 10% ruler ticks. The fill is solid graphite. The dominant mechanism's fill is pen, and its name is underlined with a 2px pen line offset 5px, so dominance never depends on color alone. All three bars share one scale, stated in the lead line.

### dQ/dV Chart
Plotly on the ground color with major and minor gridlines in the two grid tokens, axes in rule, and ticks and titles at 14px muted. The measured cell is solid ink at 3px. The fresh reference is a 2px dashed fresh-graphite line. When the main peak has fallen more than 10%, a Caveat pen note sits above the top-right corner and a 2px pen arrow points at the peak flank. There is no legend and no mode bar, and the hover label is drawn on the margin column.

### Buttons
- **Shape:** Square (0px), 1px rule border, transparent ground, ink text.
- **Hover:** On the dropzone button, the border and text turn pen. Plain buttons have no hover rule in the build.
- **Focus:** A 2px pen outline offset 2px on every focus-visible element.

### Inputs / Fields
- **Select:** Ground fill, 1px rule border, 0px radius, ink text and chevron. Focus-within turns the border pen.
- **Dropzone:** Transparent, 1px dashed rule, square.
- **Menu:** Margin-column fill. Hovered or selected options use the selection wash.
- **Expander:** 1px rule border, transparent, square. The summary text turns pen on hover.

### Table
Peak features use full width, collapsed borders and tabular figures at 0.95rem. Headers are bold and muted over a 1px rule. Rows are divided by the major-grid color. Cells are padded 0.4rem by 0.5rem.

## Do's and Don'ts

### Do:
- **Do** put every new surface on the millimetre grid (10px minor, 50px major) and keep its containers transparent so the grid shows through.
- **Do** define every new color as a dark/light pair with the same role.
- **Do** use the red pen only for change, warning, focus and annotation, and pair it with a non-color cue (letter, underline, stamp text).
- **Do** use 1px rules for structure, and 2px pen borders only for stamps and warning notes.
- **Do** set numbers in bold tabular figures.
- **Do** show states and caveats as bordered stamps or notes that stay on the sheet.

### Don't:
- **Don't** round any corner; the radius is 0px.
- **Don't** add shadows, card fills or tinted panels on the sheet. The margin-column tint belongs to the sidebar and menus only.
- **Don't** use Archivo for anything but the grade letter, or Caveat for anything but red-pen chart notes.
- **Don't** introduce a second accent hue or a status palette (green/amber/red). The grade is a letter, and warnings are pen.
- **Don't** convey the grade or the dominant mechanism through color alone.
