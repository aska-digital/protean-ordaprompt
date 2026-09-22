import { PALETTE_AREA, cn, host } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'
import { useEffect, useState } from 'react'

const ID = 'protean-ordaprompt'

async function runDemo(ctx) {
  const fixture = await ctx.rest('/demo')
  return ctx.rest('/route', {
    method: 'POST',
    body: { request: fixture.request, candidates: fixture.candidates, surface: 'topic' },
    timeoutMs: 15000,
  })
}

function OrdaPane({ ctx }) {
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const run = async () => {
    setBusy(true)
    setError(null)
    try {
      setResult(await runDemo(ctx))
    } catch (err) {
      setResult(null)
      setError(String(err?.message || err))
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => { run() }, [])

  const decision = result?.routing_decision
  return jsxs('div', {
    className: 'flex h-full flex-col gap-3 p-3 text-sm',
    children: [
      jsxs('div', { className: 'flex items-center justify-between', children: [
        jsx('div', { className: 'font-medium', children: 'OrdaPrompt' }),
        jsx('span', { className: 'text-(--ui-text-tertiary)', children: 'local · locked' }),
      ] }),
      jsx('div', { className: 'text-(--ui-text-secondary)', children: 'Manual routing test. It never routes a chat turn.' }),
      jsx('button', {
        className: cn('rounded px-3 py-2', 'bg-(--ui-accent) text-(--ui-text-on-accent)', 'disabled:opacity-50'),
        type: 'button', disabled: busy, onClick: run,
        children: busy ? 'Running…' : 'Run synthetic test',
      }),
      error ? jsx('pre', { className: 'whitespace-pre-wrap text-red-400', children: error }) : null,
      decision ? jsxs('div', { className: 'rounded border border-(--ui-stroke-secondary) p-2', children: [
        jsxs('div', { className: 'flex justify-between', children: [
          jsx('span', { children: 'Decision' }),
          jsx('strong', { children: decision.band || decision.outcome || 'returned' }),
        ] }),
        jsx('pre', { className: 'mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-xs text-(--ui-text-secondary)', children: JSON.stringify(result, null, 2) }),
      ] }) : null,
    ],
  })
}

export default {
  id: ID,
  name: 'OrdaPrompt',
  defaultEnabled: false,
  register(ctx) {
    ctx.register({
      id: 'manual-test',
      area: PALETTE_AREA,
      data: {
        id: 'protean-ordaprompt.manual-test',
        label: 'Run OrdaPrompt manual routing test',
        keywords: ['ordaprompt', 'route'],
        run: async () => {
          try {
            const result = await runDemo(ctx)
            const decision = result?.routing_decision?.band || 'returned'
            host.notify({ kind: 'info', message: `OrdaPrompt: ${decision} (local, calibration locked)` })
          } catch (err) {
            host.notify({ kind: 'error', message: `OrdaPrompt refused: ${String(err?.message || err)}` })
          }
        },
      },
    })
    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'OrdaPrompt',
      data: { placement: 'right', width: '360px' },
      render: () => jsx(OrdaPane, { ctx }),
    })
  },
}
