'use client'

import { useState } from 'react'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  ScatterChart, 
  Scatter,
  ReferenceLine,
  Area,
  AreaChart,
  Cell
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'

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

interface PerformanceChartProps {
  data: AlphaData[]
  cumulativeReturns?: CumulativeReturnsData
}

export function PerformanceChart({ data, cumulativeReturns }: PerformanceChartProps) {
  const [showTop, setShowTop] = useState(20)

  // Prepare different datasets
  const topPerformers = data.slice(0, showTop)
  
  const chartData = topPerformers.map((alpha, index) => ({
    name: topPerformers.length <= 30 ? alpha.name.replace(/_/g, ' ') : alpha.name.replace(/_/g, ' ').substring(0, 15) + '...',
    fullName: alpha.name,
    rank: index + 1,
    sharpe: Number(alpha.sharpe_ratio.toFixed(3)),
    return: Number((alpha.annualized_return * 100).toFixed(2)),
    calmar: Number(alpha.calmar_ratio.toFixed(3)),
    drawdown: Number((Math.abs(alpha.max_drawdown) * 100).toFixed(2)),
    volatility: Number((alpha.volatility * 100).toFixed(2)),
    numTrades: alpha.num_trades
  }))

  // Get top 10 strategies for highlighting
  const top10Names = data.slice(0, 10).map(alpha => alpha.name)
  
  // Create a map of strategy names to their ranking
  const rankMap = data.reduce((acc, alpha, index) => {
    acc[alpha.name] = index + 1
    return acc
  }, {} as Record<string, number>)
  
  const riskReturnData = data
    .map(alpha => ({
      name: alpha.name,
      risk: Number((alpha.volatility * 100).toFixed(2)),
      return: Number((alpha.annualized_return * 100).toFixed(2)),
      sharpe: Number(alpha.sharpe_ratio.toFixed(2)),
      drawdown: Number((Math.abs(alpha.max_drawdown) * 100).toFixed(1)),
      isTop10: top10Names.includes(alpha.name),
      rank: rankMap[alpha.name]
    }))
    .sort((a, b) => a.risk - b.risk)

  const performanceDistribution = {
    sharpe: data.reduce((acc, alpha) => {
      const bucket = Math.floor(alpha.sharpe_ratio * 2) / 2 // 0.5 buckets
      acc[bucket] = (acc[bucket] || 0) + 1
      return acc
    }, {} as Record<number, number>),
    returns: data.reduce((acc, alpha) => {
      const bucket = Math.floor(alpha.annualized_return * 20) / 20 // 0.05 buckets
      acc[bucket] = (acc[bucket] || 0) + 1
      return acc
    }, {} as Record<number, number>)
  }

  const distributionData = (() => {
    const entries = Object.entries(performanceDistribution.sharpe)
      .map(([bucket, count]) => ({
        bucket: parseFloat(bucket),
        count,
        label: `${parseFloat(bucket).toFixed(1)}`
      }))
      .sort((a, b) => a.bucket - b.bucket)
    
    // Fill missing buckets to create continuous x-axis
    const result = []
    const minBucket = Math.min(...entries.map(e => e.bucket))
    const maxBucket = Math.max(...entries.map(e => e.bucket))
    const stepSize = 0.5
    
    for (let bucket = minBucket; bucket <= maxBucket; bucket += stepSize) {
      const existing = entries.find(e => Math.abs(e.bucket - bucket) < 0.001)
      result.push({
        bucket: parseFloat(bucket.toFixed(1)),
        count: existing ? existing.count : 0,
        label: `${parseFloat(bucket.toFixed(1))}`
      })
    }
    return result
  })()

  const returnDistributionData = (() => {
    const entries = Object.entries(performanceDistribution.returns)
      .map(([bucket, count]) => ({
        bucket: parseFloat(bucket),
        count,
        label: `${(parseFloat(bucket) * 100).toFixed(0)}%`
      }))
      .sort((a, b) => a.bucket - b.bucket)
    
    // Fill missing buckets to create continuous x-axis
    const result = []
    const minBucket = Math.min(...entries.map(e => e.bucket))
    const maxBucket = Math.max(...entries.map(e => e.bucket))
    const stepSize = 0.05
    
    for (let bucket = minBucket; bucket <= maxBucket; bucket += stepSize) {
      const existing = entries.find(e => Math.abs(e.bucket - bucket) < 0.001)
      result.push({
        bucket: parseFloat(bucket.toFixed(2)),
        count: existing ? existing.count : 0,
        label: `${(parseFloat(bucket.toFixed(2)) * 100).toFixed(0)}%`
      })
    }
    return result
  })()

  const customTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold">{payload[0]?.payload?.fullName || label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.dataKey}: ${entry.value}${entry.dataKey === 'return' || entry.dataKey === 'drawdown' || entry.dataKey === 'volatility' ? '%' : ''}`}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex gap-4 items-center">
        <label className="text-sm font-medium">Show top:</label>
        <select 
          value={showTop} 
          onChange={(e) => setShowTop(Number(e.target.value))}
          className="border rounded px-3 py-1"
        >
          <option value={10}>10</option>
          <option value={20}>20</option>
          <option value={30}>30</option>
          <option value={50}>50</option>
        </select>
        <span className="text-sm text-gray-500">strategies</span>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="detailed">Detailed Metrics</TabsTrigger>
          <TabsTrigger value="risk-return">Risk vs Return</TabsTrigger>
          <TabsTrigger value="distribution">Distribution</TabsTrigger>
          <TabsTrigger value="comparison">Comparison</TabsTrigger>
          <TabsTrigger value="cumulative">Cumulative Returns</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Sharpe Ratio Rankings</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={chartData} margin={{ bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="name" 
                      angle={-45} 
                      textAnchor="end" 
                      height={topPerformers.length <= 30 ? 120 : 100}
                      fontSize={topPerformers.length <= 30 ? 9 : 10}
                      interval={0}
                    />
                    <YAxis />
                    <Tooltip content={customTooltip} />
                    <ReferenceLine y={1} stroke="red" strokeDasharray="5 5" />
                    <Bar dataKey="sharpe" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Annualized Returns (%)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={chartData} margin={{ bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="name" 
                      angle={-45} 
                      textAnchor="end" 
                      height={topPerformers.length <= 30 ? 120 : 100}
                      fontSize={topPerformers.length <= 30 ? 9 : 10}
                      interval={0}
                    />
                    <YAxis />
                    <Tooltip content={customTooltip} />
                    <ReferenceLine y={0} stroke="red" strokeDasharray="5 5" />
                    <Bar dataKey="return" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="detailed" className="space-y-6">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Calmar Ratio</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={chartData} margin={{ bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="name" 
                      angle={-45} 
                      textAnchor="end" 
                      height={topPerformers.length <= 30 ? 120 : 100}
                      fontSize={topPerformers.length <= 30 ? 9 : 10}
                      interval={0}
                    />
                    <YAxis />
                    <Tooltip content={customTooltip} />
                    <Area dataKey="calmar" stroke="#f59e0b" fill="#fed7aa" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Maximum Drawdown (%)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={chartData} margin={{ bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="name" 
                      angle={-45} 
                      textAnchor="end" 
                      height={topPerformers.length <= 30 ? 120 : 100}
                      fontSize={topPerformers.length <= 30 ? 9 : 10}
                      interval={0}
                    />
                    <YAxis />
                    <Tooltip content={customTooltip} />
                    <Bar dataKey="drawdown" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Volatility (%)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={chartData} margin={{ bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="name" 
                      angle={-45} 
                      textAnchor="end" 
                      height={topPerformers.length <= 30 ? 120 : 100}
                      fontSize={topPerformers.length <= 30 ? 9 : 10}
                      interval={0}
                    />
                    <YAxis />
                    <Tooltip content={customTooltip} />
                    <Line type="monotone" dataKey="volatility" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Multi-Metric Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={chartData.slice(0, 10)} margin={{ bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="name" 
                      angle={-45} 
                      textAnchor="end" 
                      height={topPerformers.length <= 30 ? 120 : 100}
                      fontSize={topPerformers.length <= 30 ? 9 : 10}
                      interval={0}
                    />
                    <YAxis />
                    <Tooltip content={customTooltip} />
                    <Legend />
                    <Line type="monotone" dataKey="sharpe" stroke="#3b82f6" name="Sharpe Ratio" />
                    <Line type="monotone" dataKey="calmar" stroke="#f59e0b" name="Calmar Ratio" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="risk-return" className="space-y-6">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Risk vs Return (All Strategies)</CardTitle>
              </CardHeader>
              <CardContent className="flex justify-center">
                <ResponsiveContainer width="90%" height={520}>
                  <ScatterChart data={riskReturnData} margin={{ top: 20, right: 20, bottom: 50, left: 70 }}>
                    <CartesianGrid />
                    <Legend 
                      verticalAlign="top" 
                      height={36}
                      content={() => (
                        <div className="flex justify-center gap-4 text-sm">
                          <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#3b82f6' }}></div>
                            <span>Top 10 Strategies</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#d1d5db' }}></div>
                            <span>Other Strategies</span>
                          </div>
                        </div>
                      )}
                    />
                    <XAxis 
                      dataKey="risk" 
                      name="Volatility %" 
                      unit="%" 
                      type="number"
                      domain={['dataMin - 1', 'dataMax + 1']}
                      interval={0}
                    />
                    <YAxis 
                      dataKey="return" 
                      name="Annual Return %" 
                      unit="%" 
                      domain={['dataMin - 5', 'dataMax + 5']}
                    />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      labelFormatter={() => ''}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-white p-3 border rounded shadow-lg">
                              <p className="font-semibold mb-1">{data.name}</p>
                              <p className="text-sm text-gray-600 mb-2">Rank: #{data.rank}</p>
                              <p style={{ color: '#666' }}>
                                Annual Return: {data.return.toFixed(1)}%
                              </p>
                              <p style={{ color: '#666' }}>
                                Max Drawdown: {data.drawdown.toFixed(1)}%
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <ReferenceLine x={0} stroke="gray" strokeDasharray="2 2" />
                    <ReferenceLine y={0} stroke="gray" strokeDasharray="2 2" />
                    <Scatter dataKey="return">
                      {riskReturnData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.isTop10 ? "#3b82f6" : "#d1d5db"} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Drawdown vs Return</CardTitle>
              </CardHeader>
              <CardContent className="flex justify-center">
                <ResponsiveContainer width="90%" height={520}>
                  <ScatterChart data={riskReturnData} margin={{ top: 20, right: 20, bottom: 50, left: 70 }}>
                    <CartesianGrid />
                    <Legend 
                      verticalAlign="top" 
                      height={36}
                      content={() => (
                        <div className="flex justify-center gap-4 text-sm">
                          <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#dc2626' }}></div>
                            <span>Top 10 Strategies</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#d1d5db' }}></div>
                            <span>Other Strategies</span>
                          </div>
                        </div>
                      )}
                    />
                    <XAxis 
                      dataKey="drawdown" 
                      name="Max Drawdown %" 
                      unit="%" 
                      type="number"
                      domain={['dataMin - 1', 'dataMax + 1']}
                      interval={0}
                    />
                    <YAxis 
                      dataKey="return" 
                      name="Annual Return %" 
                      unit="%" 
                      domain={['dataMin - 5', 'dataMax + 5']}
                    />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      labelFormatter={() => ''}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-white p-3 border rounded shadow-lg">
                              <p className="font-semibold mb-1">{data.name}</p>
                              <p className="text-sm text-gray-600 mb-2">Rank: #{data.rank}</p>
                              <p style={{ color: '#666' }}>
                                Annual Return: {data.return.toFixed(1)}%
                              </p>
                              <p style={{ color: '#666' }}>
                                Max Drawdown: {data.drawdown.toFixed(1)}%
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <ReferenceLine y={0} stroke="gray" strokeDasharray="2 2" />
                    <Scatter dataKey="return">
                      {riskReturnData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.isTop10 ? "#dc2626" : "#d1d5db"} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="distribution" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Sharpe Ratio Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={distributionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Return Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={returnDistributionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="comparison" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Performance Heatmap (Top {Math.min(showTop, 15)} Strategies)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Strategy</th>
                      <th className="text-center p-2">Sharpe</th>
                      <th className="text-center p-2">Return %</th>
                      <th className="text-center p-2">Calmar</th>
                      <th className="text-center p-2">Drawdown %</th>
                      <th className="text-center p-2">Volatility %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chartData.slice(0, 15).map((item, index) => (
                      <tr key={index} className="border-b hover:bg-gray-50">
                        <td className="p-2 font-medium">{item.fullName.replace(/_/g, ' ')}</td>
                        <td className={`text-center p-2 ${
                          item.sharpe > 1 ? 'bg-green-100 text-green-800' : 
                          item.sharpe > 0.5 ? 'bg-yellow-100 text-yellow-800' : 
                          'bg-red-100 text-red-800'
                        }`}>
                          {item.sharpe}
                        </td>
                        <td className={`text-center p-2 ${
                          item.return > 10 ? 'bg-green-100 text-green-800' : 
                          item.return > 0 ? 'bg-yellow-100 text-yellow-800' : 
                          'bg-red-100 text-red-800'
                        }`}>
                          {item.return}%
                        </td>
                        <td className={`text-center p-2 ${
                          item.calmar > 1 ? 'bg-green-100 text-green-800' : 
                          item.calmar > 0.5 ? 'bg-yellow-100 text-yellow-800' : 
                          'bg-red-100 text-red-800'
                        }`}>
                          {item.calmar}
                        </td>
                        <td className={`text-center p-2 ${
                          item.drawdown < 10 ? 'bg-green-100 text-green-800' : 
                          item.drawdown < 20 ? 'bg-yellow-100 text-yellow-800' : 
                          'bg-red-100 text-red-800'
                        }`}>
                          {item.drawdown}%
                        </td>
                        <td className="text-center p-2">{item.volatility}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="cumulative" className="space-y-6">
          {cumulativeReturns ? (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Cumulative Returns - Top 10 Strategies</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={700}>
                    <LineChart 
                      data={cumulativeReturns.dates.map((date, index) => {
                        const dataPoint: any = { date }
                        Object.values(cumulativeReturns.strategies).forEach(strategy => {
                          dataPoint[strategy.name] = strategy.cumulative_returns[index] || 0
                        })
                        return dataPoint
                      })}
                      margin={{ bottom: 80, right: 30 }}
                    >
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
                        formatter={(value: any, name: string) => [
                          `${Number(value).toFixed(2)}%`,
                          name.replace(/_/g, ' ')
                        ]}
                      />
                      <Legend />
                      <ReferenceLine y={0} stroke="gray" strokeDasharray="2 2" />
                      {Object.values(cumulativeReturns.strategies).map((strategy, index) => {
                        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1']
                        return (
                          <Line
                            key={strategy.name}
                            type="monotone"
                            dataKey={strategy.name}
                            stroke={colors[index % colors.length]}
                            strokeWidth={2}
                            dot={false}
                            name={strategy.name.replace(/_/g, ' ')}
                          />
                        )
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Final Returns Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                    {Object.values(cumulativeReturns.strategies)
                      .sort((a, b) => b.final_return - a.final_return)
                      .map((strategy, index) => (
                        <div key={strategy.name} className="text-center p-4 border rounded-lg">
                          <div className="text-sm font-medium text-gray-600 mb-1">
                            #{index + 1}
                          </div>
                          <div className="text-xs text-gray-500 mb-2 truncate" title={strategy.name}>
                            {strategy.name.replace(/_/g, ' ')}
                          </div>
                          <div className={`text-lg font-bold ${
                            strategy.final_return > 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {strategy.final_return > 0 ? '+' : ''}{strategy.final_return.toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            Sharpe: {strategy.sharpe_ratio.toFixed(2)}
                          </div>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <p className="text-gray-500">Cumulative returns data not available</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}