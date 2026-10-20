/**
 * Figure renderer for the TileLang tutorial.
 *
 * Every figure on `/` is produced from a small JSON spec (see
 * `scripts/tutorial/SPEC.md`) so that all ~200 diagrams share one visual
 * language: the same palette (DaisyUI semantic tokens, so both themes work),
 * the same node shapes, the same arrow heads and the same typography.
 *
 * The renderer never emits colours directly. It emits class names such as
 * `tlfig-node--accent`, and `src/styles/tutorial.css` maps those to theme
 * tokens. That is what keeps a light/dark switch from needing a second SVG.
 */

const FONT_SIZE = 15;
const SMALL_SIZE = 12.5;
const LINE_HEIGHT = 20;
const SMALL_LINE_HEIGHT = 16;
const NODE_PAD_X = 16;
const NODE_PAD_Y = 12;
const NODE_MIN_W = 120;
const NODE_MAX_W = 340;
const LEVEL_GAP = 90;
const NODE_GAP = 26;
const MARGIN = 24;

export function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

/**
 * Display width of a string in em units. CJK glyphs are full width, Latin and
 * digits roughly half, so this is close enough to pick line breaks by hand.
 */
function textWidth(text) {
  let width = 0;
  for (const char of text) {
    const code = char.codePointAt(0) ?? 0;
    if (code >= 0x2e80 && code <= 0x9fff) width += 1;
    else if (code >= 0xf900 && code <= 0xfaff) width += 1;
    else if (code >= 0xff00 && code <= 0xff60) width += 1;
    else if (char === ' ') width += 0.3;
    else width += 0.56;
  }
  return width;
}

/** Wrap text to at most `maxEm` display em, honouring explicit `\n`. */
export function wrap(text, maxEm) {
  const lines = [];
  for (const paragraph of String(text).split('\n')) {
    let current = '';
    let width = 0;
    for (const char of [...paragraph]) {
      const charWidth = textWidth(char);
      if (width + charWidth > maxEm && current !== '') {
        lines.push(current);
        current = char;
        width = charWidth;
      } else {
        current += char;
        width += charWidth;
      }
    }
    lines.push(current);
  }
  return lines.length > 0 ? lines : [''];
}

function linesToWidth(lines, size) {
  const longest = lines.reduce((max, line) => Math.max(max, textWidth(line)), 0);
  return longest * size;
}

function textBlock(lines, x, y, options = {}) {
  const {
    size = FONT_SIZE,
    lineHeight = LINE_HEIGHT,
    anchor = 'middle',
    className = 'tlfig-label',
    mono = false,
  } = options;
  const start = y - ((lines.length - 1) * lineHeight) / 2;
  const parts = lines.map((line, index) => {
    const yy = start + index * lineHeight;
    const classes = mono ? `${className} tlfig-mono` : className;
    return `<text class="${classes}" x="${round(x)}" y="${round(yy)}" text-anchor="${anchor}" font-size="${size}">${escapeXml(line)}</text>`;
  });
  return parts.join('');
}

function round(value) {
  return Math.round(value * 10) / 10;
}

function measureNode(node, maxEm) {
  const label = wrap(node.label ?? '', maxEm);
  const sub = node.sub === undefined ? [] : wrap(node.sub, maxEm * 1.05);
  const code = node.code === undefined ? [] : wrap(node.code, maxEm * 1.15);
  const labelWidth = linesToWidth(label, FONT_SIZE);
  const subWidth = linesToWidth(sub, SMALL_SIZE);
  const codeWidth = linesToWidth(code, SMALL_SIZE);
  const width = Math.min(
    NODE_MAX_W,
    Math.max(NODE_MIN_W, labelWidth, subWidth, codeWidth) + NODE_PAD_X * 2,
  );
  const height =
    NODE_PAD_Y * 2 +
    label.length * LINE_HEIGHT +
    (sub.length > 0 ? sub.length * SMALL_LINE_HEIGHT + 4 : 0) +
    (code.length > 0 ? code.length * SMALL_LINE_HEIGHT + 6 : 0);
  return { label, sub, code, width, height };
}

function renderNode(node, box) {
  const { tone = 'default' } = node;
  const rx = node.shape === 'pill' ? box.height / 2 : 10;
  const shape =
    node.shape === 'diamond'
      ? `<polygon class="tlfig-node tlfig-node--${tone}" points="${[
          `${round(box.x + box.width / 2)},${round(box.y)}`,
          `${round(box.x + box.width)},${round(box.y + box.height / 2)}`,
          `${round(box.x + box.width / 2)},${round(box.y + box.height)}`,
          `${round(box.x)},${round(box.y + box.height / 2)}`,
        ].join(' ')}" />`
      : `<rect class="tlfig-node tlfig-node--${tone}" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" rx="${round(rx)}" />`;

  let cursor = box.y + NODE_PAD_Y + LINE_HEIGHT / 2;
  const parts = [shape];
  parts.push(
    textBlock(box.label, box.x + box.width / 2, cursor, {
      className: `tlfig-label tlfig-label--${tone}`,
    }),
  );
  cursor += LINE_HEIGHT;
  if (box.sub.length > 0) {
    parts.push(
      textBlock(box.sub, box.x + box.width / 2, cursor + SMALL_LINE_HEIGHT / 2 + 4, {
        size: SMALL_SIZE,
        lineHeight: SMALL_LINE_HEIGHT,
        className: 'tlfig-sub',
      }),
    );
    cursor += box.sub.length * SMALL_LINE_HEIGHT + 4;
  }
  if (box.code.length > 0) {
    parts.push(
      textBlock(box.code, box.x + box.width / 2, cursor + SMALL_LINE_HEIGHT / 2 + 6, {
        size: SMALL_SIZE,
        lineHeight: SMALL_LINE_HEIGHT,
        className: 'tlfig-code',
        mono: true,
      }),
    );
  }
  if (node.badge !== undefined) {
    const width = textWidth(node.badge) * SMALL_SIZE + 16;
    parts.push(
      `<rect class="tlfig-badge" x="${round(box.x + box.width - width)}" y="${round(box.y - 10)}" width="${round(width)}" height="20" rx="10" />`,
      `<text class="tlfig-badge-label" x="${round(box.x + box.width - width / 2)}" y="${round(box.y + 4.5)}" text-anchor="middle" font-size="${SMALL_SIZE}">${escapeXml(node.badge)}</text>`,
    );
  }
  return { markup: parts.join(''), box };
}

function arrowPath(from, to, direction, label) {
  const horizontal = direction === 'LR';
  const startX = horizontal ? from.x + from.width : from.x + from.width / 2;
  const startY = horizontal ? from.y + from.height / 2 : from.y + from.height;
  const endX = horizontal ? to.x : to.x + to.width / 2;
  const endY = horizontal ? to.y + to.height / 2 : to.y;
  const aligned = Math.abs(startX - endX) < 0.5 || Math.abs(startY - endY) < 0.5;
  const path = aligned
    ? `M ${round(startX)} ${round(startY)} L ${round(endX)} ${round(endY)}`
    : horizontal
      ? `M ${round(startX)} ${round(startY)} H ${round(endX)} V ${round(endY)}`
      : `M ${round(startX)} ${round(startY)} V ${round(endY)} H ${round(endX)}`;
  const labelX = (startX + endX) / 2;
  const labelY = (startY + endY) / 2;
  const labelMarkup =
    label === undefined
      ? ''
      : `<text class="tlfig-edge-label" x="${round(labelX + (horizontal ? 0 : 6))}" y="${round(labelY - (horizontal ? 6 : 0))}" text-anchor="${horizontal ? 'middle' : 'start'}" font-size="${SMALL_SIZE}">${escapeXml(label)}</text>`;
  return { path, labelMarkup };
}

/** Longest-path layering over the DAG, falling back to insertion order. */
function assignLevels(nodes, edges) {
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const incoming = new Map(nodes.map((node) => [node.id, []]));
  const outgoing = new Map(nodes.map((node) => [node.id, []]));
  for (const edge of edges) {
    if (!index.has(edge.from) || !index.has(edge.to)) continue;
    incoming.get(edge.to).push(edge.from);
    outgoing.get(edge.from).push(edge.to);
  }
  const level = new Map();
  const visit = (id, seen) => {
    if (level.has(id)) return level.get(id);
    if (seen.has(id)) return 0; // cycle guard
    seen.add(id);
    const parents = incoming.get(id) ?? [];
    const value = parents.length === 0 ? 0 : Math.max(...parents.map((p) => visit(p, seen))) + 1;
    seen.delete(id);
    level.set(id, value);
    return value;
  };
  nodes.forEach((node) => visit(node.id, new Set()));
  return level;
}

function layoutFlow(nodes, edges, direction, options) {
  const levels = assignLevels(nodes, edges);
  const maxLevel = Math.max(0, ...nodes.map((node) => levels.get(node.id) ?? 0));
  const groups = new Map();
  for (const node of nodes) {
    const level = levels.get(node.id) ?? 0;
    if (!groups.has(level)) groups.set(level, []);
    groups.get(level).push(node);
  }

  const nodeMaxEm = options.nodeMaxEm ?? 22;
  const measured = new Map();
  for (const node of nodes) measured.set(node.id, measureNode(node, nodeMaxEm));

  const horizontal = direction === 'LR';
  const levelSizes = [];
  for (let level = 0; level <= maxLevel; level += 1) {
    const items = groups.get(level) ?? [];
    const extent = items.reduce(
      (sum, node) =>
        sum + (horizontal ? measured.get(node.id).height : measured.get(node.id).width),
      0,
    );
    levelSizes.push(extent + Math.max(0, items.length - 1) * NODE_GAP);
  }
  const crossExtent = Math.max(options.minCross ?? 0, ...levelSizes, 0);

  const boxes = new Map();
  let mainCursor = MARGIN;
  for (let level = 0; level <= maxLevel; level += 1) {
    const items = groups.get(level) ?? [];
    const mainSize = items.reduce(
      (max, node) =>
        Math.max(max, horizontal ? measured.get(node.id).width : measured.get(node.id).height),
      0,
    );
    let crossCursor = MARGIN + (crossExtent - levelSizes[level]) / 2;
    for (const node of items) {
      const size = measured.get(node.id);
      const cross = horizontal ? size.height : size.width;
      const box = horizontal
        ? {
            x: mainCursor,
            y: crossCursor,
            width: size.width,
            height: size.height,
            label: size.label,
            sub: size.sub,
            code: size.code,
          }
        : {
            x: crossCursor,
            y: mainCursor,
            width: size.width,
            height: size.height,
            label: size.label,
            sub: size.sub,
            code: size.code,
          };
      boxes.set(node.id, box);
      crossCursor += cross + NODE_GAP;
    }
    mainCursor += mainSize + LEVEL_GAP;
  }

  const width = horizontal ? mainCursor - LEVEL_GAP + MARGIN : crossExtent + MARGIN * 2;
  const height = horizontal ? crossExtent + MARGIN * 2 : mainCursor - LEVEL_GAP + MARGIN;
  return { boxes, width, height };
}

function renderFlow(spec) {
  const nodes = spec.nodes ?? [];
  const edges = spec.edges ?? [];
  const direction = spec.direction === 'TB' ? 'TB' : 'LR';
  const { boxes, width, height } = layoutFlow(nodes, edges, direction, spec);
  const groups = renderGroups(spec.groups ?? [], boxes);
  const nodeMarkup = nodes.map((node) => renderNode(node, boxes.get(node.id)).markup).join('');
  const edgeMarkup = edges
    .map((edge) => {
      const from = boxes.get(edge.from);
      const to = boxes.get(edge.to);
      if (from === undefined || to === undefined) return '';
      const { path, labelMarkup } = arrowPath(from, to, direction, edge.label);
      const dashed = edge.dashed === true ? ' tlfig-edge--dashed' : '';
      const tone = edge.tone === undefined ? '' : ` tlfig-edge--${edge.tone}`;
      const marker = `url(#${spec.id}-arrow)`;
      return `<path class="tlfig-edge${dashed}${tone}" d="${path}" marker-end="${marker}" />${labelMarkup}`;
    })
    .join('');
  /*
    A subgraph box is drawn around its members with padding plus room for the
    label, which can push it above the first row of nodes (negative y) and clip
    the label. Shift the whole drawing instead of shrinking the box.
  */
  const shiftX = Math.max(0, Number.isFinite(groups.minX) ? -groups.minX : 0);
  const shiftY = Math.max(0, Number.isFinite(groups.minY) ? -groups.minY : 0);
  const body = groups.markup + edgeMarkup + nodeMarkup;
  const shifted =
    shiftX === 0 && shiftY === 0
      ? body
      : `<g transform="translate(${round(shiftX)} ${round(shiftY)})">${body}</g>`;
  return {
    width: width + shiftX,
    height: height + shiftY,
    body: shifted,
    notes: spec.notes ?? [],
  };
}

/** Subgraph boxes drawn behind the flow: a labelled rounded rectangle. */
function renderGroups(groups, boxes) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  const markup = groups
    .map((group) => {
      const members = (group.nodes ?? [])
        .map((id) => boxes.get(id))
        .filter((box) => box !== undefined);
      if (members.length === 0) return '';
      const pad = 18;
      const left = Math.min(...members.map((b) => b.x)) - pad;
      const top = Math.min(...members.map((b) => b.y)) - pad - 16;
      const right = Math.max(...members.map((b) => b.x + b.width)) + pad;
      const bottom = Math.max(...members.map((b) => b.y + b.height)) + pad;
      minX = Math.min(minX, left);
      minY = Math.min(minY, top);
      const tone = group.tone === undefined ? 'default' : group.tone;
      return (
        `<rect class="tlfig-group tlfig-group--${tone}" x="${round(left)}" y="${round(top)}" width="${round(right - left)}" height="${round(bottom - top)}" rx="14" />` +
        `<text class="tlfig-group-label" x="${round(left + 12)}" y="${round(top + 15)}" font-size="${SMALL_SIZE}">${escapeXml(group.label ?? '')}</text>`
      );
    })
    .join('');
  return { markup, minX, minY };
}

function renderLayers(spec) {
  const items = spec.layers ?? [];
  const width = spec.width ?? 1000;
  const bandHeight = spec.bandHeight ?? 62;
  const gap = spec.gap ?? 12;
  const parts = [];
  let y = MARGIN;
  for (const item of items) {
    const tone = item.tone ?? 'default';
    const textWidthLimit = (width - 2 * MARGIN - 220) / 15;
    const lines = wrap(item.label ?? '', textWidthLimit);
    const subLines = item.sub === undefined ? [] : wrap(item.sub, textWidthLimit * 1.1);
    const labelHeight = lines.length * LINE_HEIGHT;
    const subHeight = subLines.length * SMALL_LINE_HEIGHT;
    const contentHeight = labelHeight + (subHeight > 0 ? 6 + subHeight : 0);
    const height = Math.max(bandHeight, NODE_PAD_Y * 2 + contentHeight);
    parts.push(
      `<rect class="tlfig-node tlfig-node--${tone}" x="${MARGIN}" y="${round(y)}" width="${width - MARGIN * 2}" height="${round(height)}" rx="12" />`,
    );
    // Each text block is drawn centred on the y we pass, so compute the two
    // block centres from the top of the content instead of nudging a cursor.
    const top = y + (height - contentHeight) / 2;
    parts.push(textBlock(lines, MARGIN + 20, top + labelHeight / 2, { anchor: 'start' }));
    if (subLines.length > 0) {
      const subTop = top + labelHeight + 6;
      parts.push(
        textBlock(subLines, MARGIN + 20, subTop + subHeight / 2, {
          anchor: 'start',
          size: SMALL_SIZE,
          lineHeight: SMALL_LINE_HEIGHT,
          className: 'tlfig-sub',
        }),
      );
    }
    if (item.right !== undefined) {
      parts.push(
        textBlock(wrap(item.right, 22), width - MARGIN - 20, y + height / 2, {
          anchor: 'end',
          size: SMALL_SIZE,
          className: 'tlfig-sub',
        }),
      );
    }
    y += height + gap;
  }
  return { width, height: y - gap + MARGIN, body: parts.join(''), notes: spec.notes ?? [] };
}

function renderSteps(spec) {
  const items = spec.steps ?? [];
  const width = spec.width ?? 1000;
  const vertical = spec.direction === 'TB';
  const parts = [];
  const notes = spec.notes ?? [];
  if (vertical) {
    const height =
      items.reduce((sum, item) => sum + (item.sub === undefined ? 52 : 72) + 18, 0) + MARGIN;
    let y = MARGIN;
    items.forEach((item, index) => {
      const tone = item.tone ?? 'default';
      const boxHeight = item.sub === undefined ? 52 : 72;
      parts.push(
        `<rect class="tlfig-node tlfig-node--${tone}" x="${MARGIN}" y="${round(y)}" width="${width - MARGIN * 2}" height="${boxHeight}" rx="10" />`,
        textBlock([`${index + 1}. ${item.label ?? ''}`], MARGIN + 18, y + 22, { anchor: 'start' }),
      );
      if (item.sub !== undefined) {
        parts.push(
          textBlock(wrap(item.sub, (width - MARGIN * 2 - 60) / 12.5), MARGIN + 18, y + 50, {
            anchor: 'start',
            size: SMALL_SIZE,
            lineHeight: SMALL_LINE_HEIGHT,
            className: 'tlfig-sub',
          }),
        );
      }
      if (index < items.length - 1) {
        parts.push(
          `<path class="tlfig-edge" d="M ${round(width / 2)} ${round(y + boxHeight)} V ${round(y + boxHeight + 18)}" marker-end="url(#${spec.id}-arrow)" />`,
        );
      }
      y += boxHeight + 18;
    });
    return { width, height, body: parts.join(''), notes };
  }

  const count = Math.max(items.length, 1);
  const gap = 22;
  const boxWidth = (width - MARGIN * 2 - gap * (count - 1)) / count;
  items.forEach((item, index) => {
    const x = MARGIN + index * (boxWidth + gap);
    const tone = item.tone ?? 'default';
    const height = 96;
    parts.push(
      `<rect class="tlfig-node tlfig-node--${tone}" x="${round(x)}" y="${MARGIN}" width="${round(boxWidth)}" height="${height}" rx="10" />`,
      textBlock([`${index + 1}`], x + boxWidth / 2, MARGIN + 20, {
        size: SMALL_SIZE,
        className: 'tlfig-step-index',
      }),
      textBlock(wrap(item.label ?? '', (boxWidth - 24) / 15), x + boxWidth / 2, MARGIN + 48),
    );
    if (item.sub !== undefined) {
      parts.push(
        textBlock(wrap(item.sub, (boxWidth - 24) / 12.5), x + boxWidth / 2, MARGIN + 72, {
          size: SMALL_SIZE,
          lineHeight: SMALL_LINE_HEIGHT,
          className: 'tlfig-sub',
        }),
      );
    }
    if (index < items.length - 1) {
      parts.push(
        `<path class="tlfig-edge" d="M ${round(x + boxWidth)} ${MARGIN + 48} H ${round(x + boxWidth + gap)}" marker-end="url(#${spec.id}-arrow)" />`,
      );
    }
  });
  return { width, height: MARGIN * 2 + 96, body: parts.join(''), notes };
}

function renderCompare(spec) {
  const columns = spec.columns ?? [];
  const rows = spec.rows ?? [];
  const width = spec.width ?? 1000;
  const labelWidth = spec.labelWidth ?? 190;
  const colWidth = (width - MARGIN * 2 - labelWidth) / Math.max(columns.length, 1);
  const rowHeight = spec.rowHeight ?? 52;
  const parts = [];
  columns.forEach((column, index) => {
    const x = MARGIN + labelWidth + index * colWidth;
    parts.push(
      `<rect class="tlfig-node tlfig-node--header" x="${round(x + 6)}" y="${MARGIN}" width="${round(colWidth - 12)}" height="42" rx="10" />`,
      textBlock(wrap(column, (colWidth - 24) / 15), x + colWidth / 2, MARGIN + 21),
    );
  });
  let y = MARGIN + 42 + 10;
  for (const row of rows) {
    const values = row.values ?? [];
    const lines = values.map((value) => wrap(value ?? '', (colWidth - 24) / 13.5));
    const maxLines = Math.max(1, ...lines.map((entry) => entry.length));
    const height = Math.max(rowHeight, 22 + (maxLines - 1) * SMALL_LINE_HEIGHT + 16);
    parts.push(
      `<rect class="tlfig-row" x="${MARGIN}" y="${round(y)}" width="${width - MARGIN * 2}" height="${round(height)}" rx="8" />`,
      textBlock(wrap(row.label ?? '', (labelWidth - 28) / 15), MARGIN + 14, y + height / 2, {
        anchor: 'start',
      }),
    );
    lines.forEach((entry, index) => {
      const x = MARGIN + labelWidth + index * colWidth;
      parts.push(
        textBlock(entry, x + colWidth / 2, y + height / 2, {
          size: SMALL_SIZE,
          lineHeight: SMALL_LINE_HEIGHT,
          className: 'tlfig-sub',
        }),
      );
    });
    y += height + 6;
  }
  return { width, height: y - 6 + MARGIN, body: parts.join(''), notes: spec.notes ?? [] };
}

function renderTimeline(spec) {
  const items = spec.items ?? [];
  const width = spec.width ?? 1000;
  const axisY = spec.axisY ?? 150;
  const parts = [`<path class="tlfig-axis" d="M ${MARGIN} ${axisY} H ${width - MARGIN}" />`];
  const count = Math.max(items.length, 1);
  const step = (width - MARGIN * 2) / count;
  items.forEach((item, index) => {
    const x = MARGIN + step * (index + 0.5);
    const above = index % 2 === 0;
    const tone = item.tone ?? 'accent';
    parts.push(
      `<circle class="tlfig-dot tlfig-dot--${tone}" cx="${round(x)}" cy="${axisY}" r="6" />`,
      textBlock(wrap(item.label ?? '', step / 15 - 1), x, above ? axisY - 54 : axisY + 58),
    );
    if (item.sub !== undefined) {
      parts.push(
        textBlock(wrap(item.sub, step / 12.5 - 1), x, above ? axisY - 30 : axisY + 82, {
          size: SMALL_SIZE,
          lineHeight: SMALL_LINE_HEIGHT,
          className: 'tlfig-sub',
        }),
      );
    }
    parts.push(
      `<path class="tlfig-axis" d="M ${round(x)} ${axisY} V ${round(above ? axisY - 16 : axisY + 16)}" />`,
    );
    if (item.at !== undefined) {
      parts.push(
        textBlock([item.at], x, above ? axisY + 20 : axisY - 18, {
          size: SMALL_SIZE,
          className: 'tlfig-sub',
        }),
      );
    }
  });
  return { width, height: 240, body: parts.join(''), notes: spec.notes ?? [] };
}

function renderBars(spec) {
  const items = spec.items ?? [];
  const width = spec.width ?? 1000;
  const labelWidth = spec.labelWidth ?? 240;
  const max = spec.max ?? Math.max(...items.map((item) => item.value ?? 0), 1);
  const barHeight = 26;
  const rowGap = 16;
  const parts = [];
  let y = MARGIN;
  for (const item of items) {
    const tone = item.tone ?? 'accent';
    const value = item.value ?? 0;
    const available = width - MARGIN * 2 - labelWidth - 120;
    const barWidth = Math.max(4, (value / max) * available);
    parts.push(
      textBlock(wrap(item.label ?? '', (labelWidth - 20) / 14), MARGIN, y + barHeight / 2, {
        anchor: 'start',
        size: 14,
      }),
      `<rect class="tlfig-track" x="${round(MARGIN + labelWidth)}" y="${round(y)}" width="${round(available)}" height="${barHeight}" rx="6" />`,
      `<rect class="tlfig-bar tlfig-bar--${tone}" x="${round(MARGIN + labelWidth)}" y="${round(y)}" width="${round(barWidth)}" height="${barHeight}" rx="6" />`,
      textBlock(
        [item.display ?? String(value)],
        MARGIN + labelWidth + available + 12,
        y + barHeight / 2,
        {
          anchor: 'start',
          size: SMALL_SIZE,
          className: 'tlfig-sub',
        },
      ),
    );
    y += barHeight + rowGap;
  }
  return { width, height: y - rowGap + MARGIN, body: parts.join(''), notes: spec.notes ?? [] };
}

function renderGrid(spec) {
  const rows = spec.rows ?? [];
  const headers = spec.headers ?? [];
  const width = spec.width ?? 1000;
  const maxCells = Math.max(...rows.map((row) => (row.cells ?? []).length), 0);
  const dataColumns = Math.max(headers.length, maxCells, 1);
  const hasRowLabels = rows.some((row) => row.label !== undefined);
  // Some specs put the row-label heading in `headers[0]`; others start the
  // headers at the first data column. Detect both and keep columns aligned.
  const headersIncludeLabel = hasRowLabels && headers.length > maxCells;
  const headerOffset = hasRowLabels && !headersIncludeLabel ? 1 : 0;
  const colCount = dataColumns + headerOffset;
  const cellWidth = (width - MARGIN * 2) / colCount;
  const cellHeight = spec.cellHeight ?? 52;
  const parts = [];
  let y = MARGIN;
  if (headers.length > 0) {
    headers.forEach((header, index) => {
      const x = MARGIN + (index + headerOffset) * cellWidth;
      parts.push(
        `<rect class="tlfig-node tlfig-node--header" x="${round(x + 3)}" y="${round(y)}" width="${round(cellWidth - 6)}" height="${cellHeight - 8}" rx="8" />`,
        textBlock(
          wrap(header, (cellWidth - 20) / 14),
          x + cellWidth / 2,
          y + (cellHeight - 8) / 2,
          {
            size: SMALL_SIZE,
          },
        ),
      );
    });
    y += cellHeight;
  }
  for (const row of rows) {
    const label = row.label;
    const offset = label === undefined ? 0 : cellWidth;
    if (label !== undefined) {
      parts.push(
        `<rect class="tlfig-node tlfig-node--muted" x="${MARGIN + 3}" y="${round(y + 4)}" width="${round(cellWidth - 6)}" height="${cellHeight - 8}" rx="8" />`,
        textBlock(wrap(label, (cellWidth - 20) / 14), MARGIN + cellWidth / 2, y + cellHeight / 2, {
          size: SMALL_SIZE,
        }),
      );
    }
    (row.cells ?? []).forEach((cell, index) => {
      const x = MARGIN + offset + index * cellWidth;
      const tone = row.tone ?? 'cell';
      const lines = wrap(cell ?? '', (cellWidth - 20) / 13.5);
      parts.push(
        `<rect class="tlfig-node tlfig-node--${tone}" x="${round(x + 3)}" y="${round(y + 4)}" width="${round(cellWidth - 6)}" height="${cellHeight - 8}" rx="8" />`,
        textBlock(lines, x + cellWidth / 2, y + cellHeight / 2, {
          size: SMALL_SIZE,
          lineHeight: SMALL_LINE_HEIGHT,
          className: 'tlfig-sub',
        }),
      );
    });
    y += cellHeight;
  }
  return {
    width,
    height: y + MARGIN - cellHeight + 8,
    body: parts.join(''),
    notes: spec.notes ?? [],
  };
}

function renderCustom(spec) {
  const width = spec.width ?? 1000;
  const height = spec.height ?? 400;
  const parts = [];
  for (const shape of spec.shapes ?? []) {
    const tone = shape.tone ?? 'default';
    const x = shape.x ?? 0;
    const y = shape.y ?? 0;
    const w = shape.width ?? 120;
    const h = shape.height ?? 44;
    if (shape.type === 'text') {
      parts.push(
        textBlock(wrap(shape.label ?? '', shape.maxEm ?? 30), x + w / 2, y + h / 2, {
          size: shape.size ?? FONT_SIZE,
        }),
      );
      continue;
    }
    if (shape.type === 'circle') {
      parts.push(
        `<circle class="tlfig-node tlfig-node--${tone}" cx="${round(x + w / 2)}" cy="${round(y + h / 2)}" r="${round(Math.min(w, h) / 2)}" />`,
      );
    } else {
      parts.push(
        `<rect class="tlfig-node tlfig-node--${tone}" x="${round(x)}" y="${round(y)}" width="${round(w)}" height="${round(h)}" rx="${shape.rx ?? 10}" />`,
      );
    }
    const labelLines = wrap(shape.label ?? '', (w - 20) / 14);
    parts.push(textBlock(labelLines, x + w / 2, y + h / 2, { size: shape.size ?? FONT_SIZE }));
  }
  for (const arrow of spec.arrows ?? []) {
    const [x1, y1, x2, y2] = arrow.points ?? [0, 0, 0, 0];
    const dashed = arrow.dashed === true ? ' tlfig-edge--dashed' : '';
    parts.push(
      `<path class="tlfig-edge${dashed}" d="M ${round(x1)} ${round(y1)} L ${round(x2)} ${round(y2)}" marker-end="url(#${spec.id}-arrow)" />`,
    );
    if (arrow.label !== undefined) {
      parts.push(
        textBlock([arrow.label], (x1 + x2) / 2, (y1 + y2) / 2 - 8, {
          size: SMALL_SIZE,
          className: 'tlfig-sub',
        }),
      );
    }
  }
  return { width, height, body: parts.join(''), notes: spec.notes ?? [] };
}

const RENDERERS = {
  flow: renderFlow,
  layers: renderLayers,
  steps: renderSteps,
  compare: renderCompare,
  timeline: renderTimeline,
  bars: renderBars,
  grid: renderGrid,
  custom: renderCustom,
};

export function renderFigure(spec) {
  const renderer = RENDERERS[spec.kind ?? 'flow'];
  if (renderer === undefined) {
    throw new Error(`unknown figure kind: ${spec.kind} (${spec.id})`);
  }
  const { width, height, body } = renderer(spec);
  const title = escapeXml(spec.title ?? spec.id);
  const defs =
    `<defs>` +
    `<marker id="${spec.id}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">` +
    `<path class="tlfig-arrowhead" d="M 0 0 L 10 5 L 0 10 z" />` +
    `</marker>` +
    `</defs>`;
  return (
    `<svg class="tlfig" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${round(width)} ${round(height)}" ` +
    `role="img" aria-label="${title}" preserveAspectRatio="xMidYMid meet" ` +
    `data-wide="${width > 900 ? 'true' : 'false'}" style="max-width:${round(width)}px;width:100%;height:auto">` +
    `<title>${title}</title>${defs}${body}</svg>`
  );
}
