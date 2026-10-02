"""
Custom CSS for the research tool.
"""

CUSTOM_CSS = """
/* ========================================
   General
   ======================================== */

footer { display: none !important; }

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* ========================================
   Header
   ======================================== */

#app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 12px;
    border-bottom: 1px solid var(--border-color-primary);
    background: var(--background-fill-primary);
    min-height: 40px;
}

.header-btn {
    min-width: 0 !important;
    width: auto !important;
    flex: 0 0 auto !important;
    min-height: 36px !important;
    padding: 6px 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    white-space: nowrap;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.header-btn:hover {
    background: var(--background-fill-secondary) !important;
}

/* Gradio adds a page navbar as soon as an app has several pages (here:
   one per interface language). The language switch in the header replaces
   it. Hidden here because gr.Navbar(visible=False) has no effect in
   Gradio 6.29 (the frontend never sees the flag). */
.nav-holder {
    display: none !important;
}

/* Language switch: plain links to the other language pages, styled as a
   small segmented control next to the header buttons. */
#language-switch {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    width: auto !important;
    padding: 0 !important;
}

#language-switch nav {
    display: flex;
    gap: 2px;
    padding: 2px;
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-md);
    font-size: 12px;
    font-weight: 600;
    line-height: 1;
}

#language-switch nav > * {
    padding: 6px 8px;
    border-radius: calc(var(--radius-md) - 2px);
    text-decoration: none;
}

#language-switch [aria-current] {
    background: var(--background-fill-secondary);
    color: var(--body-text-color);
}

#language-switch a {
    color: var(--body-text-color-subdued);
}

#language-switch a:hover {
    color: var(--body-text-color);
    background: var(--background-fill-secondary);
}

#header-title {
    flex: 1;
    text-align: center;
    margin: 0;
}

#header-title p { margin: 0; }

/* ========================================
   Sidebar (left)
   ======================================== */

#sidebar-column {
    border-right: 1px solid var(--border-color-primary);
    padding: 8px 12px !important;
    overflow-y: auto;
    max-height: calc(100vh - 60px);
    min-width: 220px !important;
    max-width: 280px !important;
}

.section-label p { margin: 4px 0 !important; font-size: 0.8rem; }

.doc-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.8rem;
}

.doc-item:hover { background: var(--background-fill-secondary); }

.doc-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* token bar */
.token-bar-container { margin: 4px 0; }
.token-text { font-size: 0.7rem; color: var(--body-text-color-subdued); margin-bottom: 2px; }
.token-bar { height: 4px; background: var(--background-fill-secondary); border-radius: 2px; }
.token-bar-fill { height: 100%; border-radius: 2px; transition: width 0.3s; }
.token-bar-fill.normal { background: var(--color-accent); }
.token-bar-fill.warning { background: #f59e0b; }
.token-bar-fill.critical { background: #ef4444; }

/* history items */
.history-item {
    padding: 6px 8px;
    border-radius: 6px;
    font-size: 0.78rem;
    cursor: pointer;
    margin: 2px 0;
}
.history-item:hover { background: var(--background-fill-secondary); }

/* ========================================
   Chat area (middle)
   ======================================== */

#chat-column {
    padding: 0 !important;
}

#chatbot {
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* input card: text field + toolbar + collapsible options */
#composer {
    flex: 0 0 auto !important;   /* equal_height row would stretch it */
    margin: 8px 12px 10px !important;
    padding: 0 !important;
    gap: 0 !important;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    background: var(--input-background-fill);
    overflow: visible;
    width: auto !important;
}

/* the text field blends into the card: no frame of its own */
#message-input,
#message-input .full-container {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
#message-input .full-container { padding: 8px 10px 0 !important; }
#message-input textarea {
    font-size: 15px !important;
    min-height: 44px !important;
    max-height: 400px !important;
    overflow-y: auto !important;
    resize: vertical !important;
}

#composer-toolbar {
    align-items: center !important;
    gap: 8px !important;
    padding: 4px 8px 8px !important;
    flex-wrap: wrap !important;
}

/* Gradio wraps form fields in a .form box with its own fill */
#composer > .form,
#composer-toolbar > .form,
#options-panel > .form,
#options-checks > .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    gap: 12px !important;
}
#composer-toolbar > .form { flex: 0 1 230px !important; }

/* compact mode selector in the toolbar */
#research-mode {
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    flex-grow: 1 !important;
}
#research-mode .wrap {
    height: 36px !important;
    min-height: 36px !important;
    border-radius: 8px !important;
    background: transparent !important;
}
#research-mode .wrap-inner,
#research-mode .secondary-wrap {
    height: 34px !important;
    padding: 0 10px !important;
}
#research-mode .secondary-wrap { padding: 0 !important; }

/* discuss + start stay together and sit on the right */
#composer-actions {
    flex: 0 0 auto !important;
    width: auto !important;
    margin-left: auto !important;
    gap: 8px !important;
    flex-wrap: nowrap !important;
}
#research-mode input {
    height: 34px !important;
    font-size: 14px !important;
}

/* all toolbar buttons: readable size, same height and radius */
#composer-toolbar button,
#result-header button {
    min-height: 36px !important;
    padding: 6px 14px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    white-space: nowrap;
    flex: 0 0 auto !important;
    width: auto !important;
}

/* "Options" and "Discuss request" are secondary: quiet until hovered.
   margin-left:auto on send pushes the action group to the right. */
#options-btn, #send-btn {
    background: transparent !important;
    border: 1px solid var(--border-color-primary) !important;
    box-shadow: none !important;
}
#options-btn:hover, #send-btn:hover {
    background: var(--background-fill-secondary) !important;
}
body[data-options-open] #options-btn {
    background: var(--color-accent-soft) !important;
    border-color: var(--color-accent) !important;
}

/* the only accent button */
#research-btn {
    background: var(--button-primary-background-fill) !important;
    color: var(--button-primary-text-color) !important;
    border: none !important;
}
#research-btn:hover {
    background: var(--button-primary-background-fill-hover) !important;
}

/* options row: closed unless the "Options" button set the body flag */
#options-panel {
    padding: 8px 12px 12px !important;
    gap: 12px !important;
    border-top: 1px solid var(--border-color-primary);
    align-items: flex-end !important;
}
body:not([data-options-open]) #options-panel {
    display: none !important;
}
#options-checks {
    gap: 2px !important;
}
#options-checks label {
    font-size: 14px !important;
}

/* ========================================
   Result panel (right)
   ======================================== */

#result-panel {
    border-left: 1px solid var(--border-color-primary);
    padding: 8px 12px !important;
}

/* the panel's column stretches its children (equal_height row):
   keep the header and download field at their natural height */
#result-header, #export-file {
    flex: 0 0 auto !important;
}
#result-header {
    align-items: center !important;
    gap: 6px !important;
}
#result-title {
    flex: 1 1 0 !important;
    width: auto !important;
    min-width: 80px !important;
}
#result-title p { margin: 0; font-size: 15px; }
.export-btn {
    background: transparent !important;
    border: 1px solid var(--border-color-primary) !important;
    box-shadow: none !important;
}
.export-btn:hover { background: var(--background-fill-secondary) !important; }

/* all 4 tab contents: scroll bar for long content */
#report-display,
#sources-display {
    max-height: calc(100vh - 200px);
    overflow-y: auto;
}

/* "History" bundles three sections; the tab scrolls as a whole */
#history-tab {
    max-height: calc(100vh - 200px);
    overflow-y: auto;
}

#report-display {
    padding: 16px;
    font-size: 0.9rem;
    line-height: 1.6;
}

#report-display h1 { font-size: 1.4rem; margin: 16px 0 8px; }
#report-display h2 { font-size: 1.2rem; margin: 14px 0 6px; }
#report-display h3 { font-size: 1.05rem; margin: 12px 0 4px; }
#report-display code { background: var(--background-fill-secondary); padding: 1px 4px; border-radius: 3px; }
#report-display pre { background: var(--background-fill-secondary); padding: 12px; border-radius: 8px; overflow-x: auto; }
#report-display blockquote { border-left: 3px solid var(--color-accent); padding-left: 12px; margin: 8px 0; }

/* progress display */
.progress-phase {
    padding: 4px 8px;
    font-size: 0.8rem;
    border-radius: 4px;
    margin: 2px 0;
}

.progress-phase.active {
    background: var(--color-accent-soft);
    font-weight: 600;
}

.progress-phase.done {
    color: var(--body-text-color-subdued);
}

.source-item {
    padding: 3px 8px;
    font-size: 0.75rem;
    border-bottom: 1px solid var(--border-color-primary);
}

.source-item a {
    color: var(--color-accent);
    text-decoration: none;
}

.source-item a:hover { text-decoration: underline; }

/* ========================================
   Hidden elements
   ======================================== */

#drop-upload, #paste-buffer {
    position: fixed !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ========================================
   Responsive
   ======================================== */

@media (max-width: 768px) {
    #sidebar-column {
        position: absolute;
        z-index: 100;
        background: var(--background-fill-primary);
        left: 0;
        top: 44px;
        bottom: 0;
        width: 260px !important;
        max-width: 260px !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.15);
    }

    #result-panel {
        position: absolute;
        z-index: 100;
        background: var(--background-fill-primary);
        right: 0;
        top: 44px;
        bottom: 0;
        width: 90vw !important;
        max-width: 500px !important;
        box-shadow: -4px 0 20px rgba(0,0,0,0.15);
    }
}

/* ========================================
   Footer
   ======================================== */

#app-footer {
    text-align: center;
    padding: 4px 12px !important;
    border-top: 1px solid var(--border-color-primary);
}

#app-footer p { margin: 0; font-size: 0.7rem; color: var(--body-text-color-subdued); }
#app-footer a { color: var(--body-text-color-subdued); text-decoration: none; }
#app-footer a:hover { text-decoration: underline; }

#export-file {
    min-height: 0 !important;
}

/* invisible buttons that must stay in the DOM (for JS triggers) */
.hidden-btn {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
}
"""
