'use client'

import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'

interface AlphaData {
  name: string
  description: string
  sharpe_ratio: number
  annualized_return: number
  calmar_ratio: number
  max_drawdown: number
  volatility: number
  num_trades: number
}

interface AlphaCardProps {
  alpha: AlphaData
  rank: number
  onViewDetail: () => void
  onViewChart: () => void
}

export function AlphaCard({ alpha, rank, onViewDetail, onViewChart }: AlphaCardProps) {
  const getPerformanceColor = (value: number, metric: string) => {
    if (metric === 'sharpe_ratio' || metric === 'calmar_ratio') {
      return value > 1 ? 'text-green-600' : value > 0.5 ? 'text-yellow-600' : 'text-red-600'
    }
    if (metric === 'annualized_return') {
      return value > 0.1 ? 'text-green-600' : value > 0 ? 'text-yellow-600' : 'text-red-600'
    }
    if (metric === 'max_drawdown') {
      return value > -0.1 ? 'text-green-600' : value > -0.2 ? 'text-yellow-600' : 'text-red-600'
    }
    return 'text-gray-600'
  }

  const getRankBadgeVariant = (rank: number): "default" | "secondary" | "destructive" | "outline" => {
    return 'default'
  }

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <Badge variant={getRankBadgeVariant(rank)}>#{rank}</Badge>
            <CardTitle className="text-lg truncate">{alpha.name.replace(/_/g, ' ')}</CardTitle>
          </div>
          <div className="flex flex-col gap-1 ml-2 flex-shrink-0">
            <button
              onClick={onViewDetail}
              className="px-2 py-0.5 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors whitespace-nowrap"
            >
              View Detail
            </button>
            <button
              onClick={onViewChart}
              className="px-2 py-0.5 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors whitespace-nowrap"
            >
              View Chart
            </button>
          </div>
        </div>
        <p className="text-sm text-gray-600 line-clamp-1">{alpha.description}</p>
      </CardHeader>
      <CardContent className="py-2">
        <div className="grid grid-cols-4 gap-2">
          <div className="text-center">
            <div className="text-sm text-gray-500">Sharpe Ratio</div>
            <div className={`text-lg font-semibold ${getPerformanceColor(alpha.sharpe_ratio, 'sharpe_ratio')}`}>
              {alpha.sharpe_ratio.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-500">Annual Return</div>
            <div className={`text-lg font-semibold ${getPerformanceColor(alpha.annualized_return, 'annualized_return')}`}>
              {(alpha.annualized_return * 100).toFixed(1)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-500">Calmar Ratio</div>
            <div className={`text-lg font-semibold ${getPerformanceColor(alpha.calmar_ratio, 'calmar_ratio')}`}>
              {alpha.calmar_ratio.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-500">Max Drawdown</div>
            <div className={`text-lg font-semibold ${getPerformanceColor(alpha.max_drawdown, 'max_drawdown')}`}>
              {(alpha.max_drawdown * 100).toFixed(1)}%
            </div>
          </div>
        </div>
        <div className="mt-1 grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-gray-500">Volatility:</span>
            <span className="ml-1 font-medium">{(alpha.volatility * 100).toFixed(1)}%</span>
          </div>
          <div>
            <span className="text-gray-500">Trades:</span>
            <span className="ml-1 font-medium">{alpha.num_trades}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}