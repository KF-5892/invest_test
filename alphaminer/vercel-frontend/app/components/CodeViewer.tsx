'use client'

import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import SyntaxHighlighter from 'react-syntax-highlighter'
import tomorrow from 'react-syntax-highlighter/dist/styles/tomorrow'

interface AlphaSourceCode {
  class_name: string
  module: string
  source_code: string
  docstring: string
}

interface CodeViewerProps {
  alpha: AlphaSourceCode
  onClose: () => void
}

export function CodeViewer({ alpha, onClose }: CodeViewerProps) {
  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl">{alpha.class_name}</CardTitle>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="secondary">{alpha.module}</Badge>
            </div>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Close
          </button>
        </div>
        {alpha.docstring && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-blue-900 mb-2">Description</h4>
            <p className="text-blue-800 text-sm">{alpha.docstring}</p>
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">Source Code</h4>
            <div className="rounded-lg overflow-hidden border">
              <SyntaxHighlighter
                language="python"
                style={tomorrow}
                customStyle={{
                  margin: 0,
                  padding: '1rem',
                  fontSize: '0.875rem',
                  lineHeight: '1.5'
                }}
                showLineNumbers
              >
                {alpha.source_code}
              </SyntaxHighlighter>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium mb-2">Implementation Notes</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• This alpha inherits from the BaseAlpha class</li>
              <li>• The calculate() method returns signals for each asset</li>
              <li>• Positive signals indicate long positions, negative indicate short</li>
              <li>• All alphas are backtested on the same dataset for fair comparison</li>
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}