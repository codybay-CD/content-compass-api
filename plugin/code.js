// Content Compass — Figma Plugin Backend (code.js)
// Runs in the Figma sandbox. Extracts text nodes from the selection
// and communicates with the UI iframe for brand voice analysis.

figma.showUI(__html__, { width: 420, height: 640, title: "Content Compass" });

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS: Tree walking & context inference
// ─────────────────────────────────────────────────────────────────────────────

/** Walk up the tree to find the nearest named component/frame parent */
function findNearestComponentParent(node) {
  let current = node.parent;
  while (current) {
    if (
      current.type === "COMPONENT" ||
      current.type === "INSTANCE" ||
      (current.type === "FRAME" && current.name && current.name.trim() !== "")
    ) {
      return current;
    }
    current = current.parent;
  }
  return null;
}

/** Walk up the tree to find the top-level frame name and page name */
function getFrameAndPageContext(node) {
  const pageName = figma.currentPage ? figma.currentPage.name : "";

  let current = node.parent;
  let topFrame = null;
  while (current && current.type !== "PAGE" && current.type !== "DOCUMENT") {
    if (
      current.parent &&
      (current.parent.type === "PAGE" || current.parent.type === "DOCUMENT")
    ) {
      topFrame = current;
      break;
    }
    current = current.parent;
  }

  const frameName = topFrame
    ? topFrame.name
    : node.parent
    ? node.parent.name
    : "";
  return { frameName, pageName };
}

/** Infer UI state from component/frame name keywords */
function inferUIState(frameName, parentName, nodeName) {
  const combined = [frameName, parentName, nodeName].join(" ").toLowerCase();
  if (
    /\bloading\b|\bprogress\b|\bspinner\b|\bposting\b|\bsending\b|\bprocessing\b|\bwaiting\b/.test(
      combined
    )
  )
    return "loading";
  if (
    /\berror\b|\bfailed\b|\bfailure\b|\binvalid\b|\bwarning\b/.test(combined)
  )
    return "error";
  if (
    /\bsuccess\b|\bcomplete\b|\bconfirm\b|\bdone\b|\bfinish\b/.test(combined)
  )
    return "success";
  if (/\bempty\b|\bzero.state\b|\bno.results\b|\bblank\b/.test(combined))
    return "empty";
  if (/\btoast\b|\bsnackbar\b|\bnotif\b|\bbanner\b/.test(combined))
    return "notification";
  if (/\bonboard\b|\bwelcome\b|\bintro\b|\bfirst.time\b/.test(combined))
    return "onboarding";
  return "default";
}

/** Infer the semantic role of a text node from its name, parent name, and font size */
function inferTextRole(node, parentNode) {
  const name = (node.name || "").toLowerCase();
  const parentName = (parentNode ? parentNode.name : "").toLowerCase();
  const chars = (node.characters || "").trim();

  // Button / CTA detection — check node name, parent name, and grandparent
  if (
    /button|cta|\bbtn\b|action|submit|confirm|cancel|\bprimary\b|\bsecondary\b|\bghost\b/.test(name) ||
    /button|cta|\bbtn\b/.test(parentName)
  )
    return "button-label";

  // Heading detection — node name, parent name, or large font size
  const fontSize = node.fontSize || 0;
  if (
    /^(title|heading|header|h1|h2|h3|display|hero|eyebrow|kicker)/.test(name) ||
    /title|heading|header|display|hero/.test(parentName) ||
    fontSize >= 20
  )
    return "heading";

  // Navigation / tab bar items
  if (/^(tab|nav|menu|item|link|breadcrumb)/.test(name) || /tab.bar|nav.bar|menu|navbar|tabbar/.test(parentName))
    return "navigation";

  // Caption / helper / subtitle
  if (/^(caption|hint|helper|subtext|subtitle|meta|label|footnote|supporting)/.test(name) || fontSize <= 11)
    return "caption";

  // Placeholder text
  if (/^(placeholder|ghost|empty.state)/.test(name))
    return "placeholder";

  // Error / warning messages
  if (/^(error|warning|alert|notice|validation)/.test(name))
    return "error-message";

  // Link text
  if (/^(link|url|anchor)/.test(name))
    return "link";

  // Badge / chip / tag
  if (/^(badge|tag|chip|pill|status|count)/.test(name))
    return "badge";

  // Body / description / paragraph
  if (/^(body|description|content|paragraph|text|copy|detail)/.test(name))
    return "body";

  // Fallback: short strings (≤ 30 chars) with no spaces are likely labels/buttons
  if (chars.length <= 30 && !chars.includes(" "))
    return "button-label";

  // Fallback: short strings (≤ 40 chars) are likely headings or captions
  if (chars.length <= 40)
    return "heading";

  return "body";
}

/**
 * Return true if a node, or any of its ancestors up to the canvas, is hidden.
 * Figma exposes visibility via the `visible` property (true = shown, false = hidden).
 */
function isNodeHidden(node) {
  let current = node;
  while (current && current.type !== "PAGE" && current.type !== "DOCUMENT") {
    if (current.visible === false) return true;
    current = current.parent;
  }
  return false;
}

/**
 * Return true if the node is not visible within the rendered frame.
 *
 * Stage 1 — absoluteRenderBounds null check: Figma sets this to null when a
 *   node is fully clipped by a parent with clipsContent=true.
 * Stage 2 — Top-level frame bounds check: zero overlap with the selected frame.
 * Stage 3 — Clipping ancestor walk: zero overlap with any clipping ancestor.
 *
 * A node is "outside" only if it has ZERO overlap — partially clipped nodes are kept.
 */
function isNodeNotVisible(node, topLevelFrameBounds) {
  if (node.absoluteRenderBounds === null) return true;

  const nb = node.absoluteBoundingBox;
  if (!nb) return false;

  if (topLevelFrameBounds) {
    const fb = topLevelFrameBounds;
    const outsideTopLevel =
      nb.x + nb.width <= fb.x ||
      nb.x >= fb.x + fb.width ||
      nb.y + nb.height <= fb.y ||
      nb.y >= fb.y + fb.height;
    if (outsideTopLevel) return true;
  }

  let ancestor = node.parent;
  while (ancestor && ancestor.type !== "PAGE" && ancestor.type !== "DOCUMENT") {
    const canClip =
      ancestor.type === "FRAME" ||
      ancestor.type === "COMPONENT" ||
      ancestor.type === "INSTANCE";

    if (canClip && ancestor.clipsContent === true) {
      const ab = ancestor.absoluteBoundingBox;
      if (ab) {
        const outsideAncestor =
          nb.x + nb.width <= ab.x ||
          nb.x >= ab.x + ab.width ||
          nb.y + nb.height <= ab.y ||
          nb.y >= ab.y + ab.height;
        if (outsideAncestor) return true;
      }
    }
    ancestor = ancestor.parent;
  }

  return false;
}

/**
 * Extract all visible, in-bounds text nodes from a node tree.
 * Hidden nodes and entire hidden subtrees are skipped.
 * Nodes outside the selected frame's rendered area are also excluded.
 */
function extractTextNodes(node, topLevelNode, topLevelFrameBounds) {
  const texts = [];

  if (isNodeHidden(node)) return texts;

  if (node.type === "TEXT") {
    if (isNodeNotVisible(node, topLevelFrameBounds)) return texts;

    const parent = findNearestComponentParent(node);
    const role = inferTextRole(node, parent);
    const { frameName, pageName } = getFrameAndPageContext(node);
    const parentName = parent
      ? parent.name
      : topLevelNode
      ? topLevelNode.name
      : "Unknown";
    const uiState = inferUIState(frameName, parentName, node.name);

    texts.push({
      id: node.id,
      name: node.name,
      characters: node.characters,
      parentName: parentName,
      parentType: parent ? parent.type : "FRAME",
      role: role,
      frameName: frameName,
      pageName: pageName,
      uiState: uiState,
    });
  }

  if ("children" in node) {
    for (const child of node.children) {
      texts.push(...extractTextNodes(child, topLevelNode, topLevelFrameBounds));
    }
  }

  return texts;
}

// ─────────────────────────────────────────────────────────────────────────────
// SELECTION HANDLING
// ─────────────────────────────────────────────────────────────────────────────

function sendSelectionToUI() {
  const selection = figma.currentPage.selection;

  if (selection.length === 0) {
    figma.ui.postMessage({ type: "NO_SELECTION" });
    return;
  }

  const selectedNode = selection[0];
  const frameBounds = selectedNode.absoluteBoundingBox || null;
  const textNodes = extractTextNodes(selectedNode, selectedNode, frameBounds);

  if (textNodes.length === 0) {
    figma.ui.postMessage({ type: "NO_TEXT", nodeName: selectedNode.name });
    return;
  }

  figma.ui.postMessage({
    type: "SELECTION_READY",
    nodeName: selectedNode.name,
    nodeType: selectedNode.type,
    textNodes: textNodes,
    totalTextNodes: textNodes.length,
  });
}

// Listen for selection changes and send initial state
figma.on("selectionchange", sendSelectionToUI);
sendSelectionToUI();

// ─────────────────────────────────────────────────────────────────────────────
// MESSAGE HANDLER
// ─────────────────────────────────────────────────────────────────────────────

figma.ui.onmessage = (msg) => {
  // Apply a rewrite to a specific text node
  if (msg.type === "APPLY_REWRITE") {
    const node = figma.getNodeById(msg.nodeId);
    if (node && node.type === "TEXT") {
      figma.loadFontAsync(node.fontName).then(() => {
        node.characters = msg.newText;
        figma.ui.postMessage({ type: "REWRITE_APPLIED", nodeId: msg.nodeId });
      });
    }
  }

  // Close the plugin
  if (msg.type === "CLOSE") {
    figma.closePlugin();
  }

  // Resize the plugin window
  if (msg.type === "RESIZE") {
    figma.ui.resize(msg.width, msg.height);
  }

  // Re-send selection info on demand
  if (msg.type === "REFRESH_SELECTION") {
    sendSelectionToUI();
  }
};
