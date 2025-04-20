# Quick Start Section Implementation Plan

## 1. High‑Level Goal
Add a new “Quick Start” section to the root homepage (`app/page.tsx`), directly below the existing sign‑in / Google‑button container.  
This section will display a Python example code snippet with syntax highlighting and a “copy” button, leveraging our existing `CodeBlock` component (extended to support languages).

## 2. Outline of Tasks
1. Extend `components/tutorial/code-block.tsx` to:
   - Accept an optional `language` prop (e.g. `"python"`).
   - Use `react-syntax-highlighter` under the hood instead of a raw `<pre><code>`, so code is highlighted.
   - Preserve the “copy” button logic.
2. Update `app/page.tsx` to:
   - Import the enhanced `CodeBlock`.
   - Insert a new `<section>` (styled via Tailwind) containing a heading and the Python snippet.
3. Style and test:
   - Ensure Tailwind classes match existing design.
   - Verify syntax highlighting loads correctly.
   - Test the copy button.

## 3. File Analysis
### A. components/tutorial/code-block.tsx
- Currently only wraps `<pre><code>` and manages copy icon.
- Needs to swap in `ReactSyntaxHighlighter` (from `react-syntax-highlighter`) when a `language` prop is provided.

### B. app/page.tsx
- Renders Hero, Google button, “Or continue with” divider, and email sign‑in link.
- We will append a Quick Start `<section>` immediately after that container’s closing `</div>`.

### C. Dependencies
- `react-syntax-highlighter` is already in `package.json`.
- No new packages are required.

## 4. Detailed Implementation

### 4.1 Extend CodeBlock
In `components/tutorial/code-block.tsx`:
1. Add imports:
   ```ts
   import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
   import python from 'react-syntax-highlighter/dist/esm/languages/hljs/python';
   import { vs2015 } from 'react-syntax-highlighter/dist/esm/styles/hljs';
   ```
2. Register language once (top‐level):
   ```ts
   SyntaxHighlighter.registerLanguage('python', python);
   ```
3. Update component signature:
   ```ts
   export function CodeBlock({
     code,
     language = ''
   }: {
     code: string;
     language?: string;
   }) { … }
   ```
4. Replace raw `<pre><code>` with:
   ```tsx
   <SyntaxHighlighter
     language={language}
     style={vs2015}
     customStyle={{ padding: 0, margin: 0, background: 'transparent' }}
     codeTagProps={{ className: 'p-3 text-xs' }}
   >
     {code}
   </SyntaxHighlighter>
   ```
5. Keep the `<Button>` and copy logic unchanged.

### 4.2 Add Quick Start Section
In `app/page.tsx`, below the existing sign‑in block:
1. Import:
   ```ts
   import { CodeBlock } from '@/components/tutorial/code-block';
   ```
2. After the `<Button asChild>` block, append:
   ```tsx
   {/* Quick Start */}
   <section className="mt-12 w-full max-w-2xl bg-card p-6 rounded-lg shadow">
     <h2 className="text-2xl font-semibold mb-4">Quick Start</h2>
     <CodeBlock
       language="python"
       code={`from lmsys import Local
import os

cwd = os.getcwd()

# Initialize the SDK…`}
     />
   </section>
   ```
   *Replace `…` with the full example from your spec.*

### 4.3 Styling & Theming
- Use existing Tailwind CSS variables (`bg-card`, `text-foreground`, etc.) to ensure dark‑mode compatibility.
- The `shadow` / `rounded-lg` classes match site cards.

## 5. Verification
1. Start dev server (`npm run dev`).
2. Visit `/` (unauthenticated).
3. Confirm:
   - New Quick Start section appears below sign‑in area.
   - Python code is syntax‑highlighted.
   - Copy button copies the snippet.
   - Dark/light theme swaps correctly.
4. Test mobile breakpoints.

Once verified, commit, open a PR, and request review.

*End of implementation plan.*  
