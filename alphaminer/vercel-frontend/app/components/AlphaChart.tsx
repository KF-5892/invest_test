'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
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

interface CumulativeReturnsData {
  dates: string[]
  strategies: Record<string, {
    name: string
    sharpe_ratio: number
    annualized_return: number
    cumulative_returns: number[]
    final_return: number
  }>
}

interface AlphaChartProps {
  alpha: AlphaData
  cumulativeReturns: CumulativeReturnsData | null
  onClose: () => void
}

export function AlphaChart({ alpha, cumulativeReturns, onClose }: AlphaChartProps) {
  const getStrategyData = () => {
    if (!cumulativeReturns) return null
    
    const strategyKey = Object.keys(cumulativeReturns.strategies).find(key => 
      cumulativeReturns.strategies[key].name === alpha.name
    )
    
    if (!strategyKey) return null
    
    const strategy = cumulativeReturns.strategies[strategyKey]
    return cumulativeReturns.dates.map((date, index) => ({
      date,
      return: strategy.cumulative_returns[index] || 0
    }))
  }

  const chartData = getStrategyData()

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

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl">{alpha.name.replace(/_/g, ' ')}</CardTitle>
              <p className="text-gray-600 mt-1">{alpha.description}</p>
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
            >
              Close
            </button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-6">
            <div className="text-center">
              <div className="text-sm text-gray-500 mb-1">Sharpe Ratio</div>
              <Badge variant="outline" className={getPerformanceColor(alpha.sharpe_ratio, 'sharpe_ratio')}>
                {alpha.sharpe_ratio.toFixed(2)}
              </Badge>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500 mb-1">Annual Return</div>
              <Badge variant="outline" className={getPerformanceColor(alpha.annualized_return, 'annualized_return')}>
                {(alpha.annualized_return * 100).toFixed(1)}%
              </Badge>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500 mb-1">Calmar Ratio</div>
              <Badge variant="outline" className={getPerformanceColor(alpha.calmar_ratio, 'calmar_ratio')}>
                {alpha.calmar_ratio.toFixed(2)}
              </Badge>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-500 mb-1">Max Drawdown</div>
              <Badge variant="outline" className={getPerformanceColor(alpha.max_drawdown, 'max_drawdown')}>
                {(alpha.max_drawdown * 100).toFixed(1)}%
              </Badge>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-6 text-sm mb-6">
            <div className="flex justify-between">
              <span className="text-gray-500">Volatility:</span>
              <span className="font-medium">{(alpha.volatility * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Number of Trades:</span>
              <span className="font-medium">{alpha.num_trades}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Cumulative Returns Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          {chartData ? (
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData} margin={{ bottom: 80, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  fontSize={10}
                  type="category"
                  tick={{ fontSize: 10 }}
                  interval="preserveStartEnd"
                  minTickGap={30}
                />
                <YAxis 
                  label={{ value: 'Cumulative Return (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  labelFormatter={(value) => `Date: ${value}`}
                  formatter={(value: any) => [`${Number(value).toFixed(2)}%`, 'Cumulative Return']}
                />
                <ReferenceLine y={0} stroke="gray" strokeDasharray="2 2" />
                <Line
                  type="monotone"
                  dataKey="return"
                  stroke="#3b82f6"
                  strokeWidth={3}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <p>Cumulative returns data not available for this strategy</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}