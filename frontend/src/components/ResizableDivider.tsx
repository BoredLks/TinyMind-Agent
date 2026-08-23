import { useCallback, useRef } from 'react'

interface Props {
  currentWidth: number
  onResize: (newWidth: number) => void
  minWidth?: number
  maxWidth?: number
  invert?: boolean // true for right-side panels: dragging right shrinks the panel
}

export default function ResizableDivider({ currentWidth, onResize, minWidth = 200, maxWidth = 500, invert = false }: Props) {
  const startX = useRef(0)
  const startW = useRef(0)

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      startX.current = e.clientX
      startW.current = currentWidth

      const onMouseMove = (ev: MouseEvent) => {
        const rawDelta = ev.clientX - startX.current
        const delta = invert ? -rawDelta : rawDelta
        const newWidth = Math.max(minWidth, Math.min(maxWidth, startW.current + delta))
        onResize(newWidth)
      }

      const onMouseUp = () => {
        document.removeEventListener('mousemove', onMouseMove)
        document.removeEventListener('mouseup', onMouseUp)
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }

      document.addEventListener('mousemove', onMouseMove)
      document.addEventListener('mouseup', onMouseUp)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    },
    [currentWidth, onResize, minWidth, maxWidth, invert],
  )

  return (
    <div
      onMouseDown={onMouseDown}
      className="w-1 cursor-col-resize shrink-0 group"
      style={{ minWidth: 4 }}
      title="拖动调整宽度"
    >
      <div className="w-0.5 h-full mx-auto bg-transparent group-hover:bg-blue-400 group-active:bg-blue-500 transition-colors" />
    </div>
  )
}