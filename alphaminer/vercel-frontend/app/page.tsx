'use client'

import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, ReferenceLine } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs'
import { Badge } from './components/ui/badge'
import { CodeViewer } from './components/CodeViewer'
import { AlphaCard } from './components/AlphaCard'
import { PerformanceChart } from './components/PerformanceChart'
import { AlphaChart } from './components/AlphaChart'

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

interface AlphaSourceCode {
  class_name: string
  module: string
  source_code: string
  docstring: string
}

export default function Home() {
  const [alphaData, setAlphaData] = useState<AlphaData[]>([])
  const [sourceCode, setSourceCode] = useState<Record<string, AlphaSourceCode>>({})
  const [cumulativeReturns, setCumulativeReturns] = useState<any>(null)
  const [selectedAlpha, setSelectedAlpha] = useState<string | null>(null)
  const [selectedAlphaChart, setSelectedAlphaChart] = useState<string | null>(null)
  const [rightPanelMode, setRightPanelMode] = useState<'detail' | 'chart' | null>(null)
  const [loading, setLoading] = useState(true)
  const [nameToClassMap, setNameToClassMap] = useState<Record<string, string>>({})

  useEffect(() => {
    const loadData = async () => {
      try {
        const [alphaResponse, sourceResponse, cumulativeResponse] = await Promise.all([
          fetch('/data/alpha_leaderboard.json'),
          fetch('/data/source_code.json'),
          fetch('/data/cumulative_returns.json').catch(() => null)
        ])
        
        const alphas = await alphaResponse.json()
        const sources = await sourceResponse.json()
        const cumulative = cumulativeResponse ? await cumulativeResponse.json() : null
        
        setAlphaData(alphas)
        setSourceCode(sources)
        setCumulativeReturns(cumulative)
        
        // Create mapping from leaderboard names to source code class names
        const mapping: Record<string, string> = {}
        Object.keys(sources).forEach(className => {
          const code = sources[className].source_code
          
          // Look for name= patterns in the source code
          const nameMatch = code.match(/name=f?["']([\w_{}d]+)["']/)
          if (nameMatch) {
            let namePattern = nameMatch[1]
            // Handle f-string patterns like 'drawdown_recovery_{lookback}d'
            if (namePattern.includes('{')) {
              // Try to match specific instances from leaderboard
              alphas.forEach((alpha: any) => {
                const regex = new RegExp('^' + namePattern.replace(/{[^}]+}/g, '\\d+(?:\\.\\d+)?[a-zA-Z]*') + '$')
                if (regex.test(alpha.name)) {
                  mapping[alpha.name] = className
                }
              })
            } else {
              mapping[namePattern] = className
            }
          }
        })
        
        setNameToClassMap(mapping)
        setLoading(false)
      } catch (error) {
        console.error('Error loading data:', error)
        setLoading(false)
      }
    }

    loadData()
  }, [])

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

  const topPerformers = alphaData.slice(0, 10)
  const chartData = topPerformers.map(alpha => ({
    name: alpha.name.replace(/_/g, ' ').substring(0, 20),
    sharpe: alpha.sharpe_ratio,
    return: alpha.annualized_return * 100,
    calmar: alpha.calmar_ratio,
    drawdown: Math.abs(alpha.max_drawdown) * 100
  }))

  // Risk vs Return data sorted by max drawdown for proper x-axis positioning
  const riskReturnChartData = topPerformers
    .map((alpha, index) => ({
      name: alpha.name.replace(/_/g, ' ').substring(0, 20),
      fullName: alpha.name,
      return: alpha.annualized_return * 100,
      drawdown: Math.abs(alpha.max_drawdown) * 100,
      rank: index + 1
    }))
    .sort((a, b) => a.drawdown - b.drawdown) // Sort by drawdown ascending (smallest to largest)

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-xl">Loading Alpha Dashboard...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Alpha Miners Dashboard</h1>
          <p className="text-gray-600">Advanced algorithmic trading strategies performance analysis</p>
        </header>

        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance Charts</TabsTrigger>
            <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
            <TabsTrigger value="source">Source Code</TabsTrigger>
            <TabsTrigger value="individual">Individual Chart</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Total Strategies</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{alphaData.length}</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Avg Sharpe Ratio</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {(alphaData.reduce((sum, a) => sum + a.sharpe_ratio, 0) / alphaData.length).toFixed(2)}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Best Performer</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-lg font-semibold text-green-600">
                    {alphaData[0]?.name.replace(/_/g, ' ')}
                  </div>
                  <div className="text-sm text-gray-500">
                    Sharpe: {alphaData[0]?.sharpe_ratio.toFixed(2)}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Positive Strategies</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {alphaData.filter(a => a.annualized_return > 0).length}
                  </div>
                  <div className="text-sm text-gray-500">
                    {((alphaData.filter(a => a.annualized_return > 0).length / alphaData.length) * 100).toFixed(0)}%
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Top 10 Strategies - Sharpe Ratio</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} fontSize={11} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="sharpe" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Risk vs Return</CardTitle>
                </CardHeader>
                <CardContent>
                  <div style={{ marginLeft: '5%' }}>
                    <ResponsiveContainer width="95%" height={300}>
                    <ScatterChart data={riskReturnChartData} margin={{ left: 20, bottom: 10 }}>
                      <CartesianGrid />
                      <XAxis 
                        dataKey="drawdown" 
                        name="Max Drawdown %" 
                        type="number"
                        domain={['dataMin - 1', 'dataMax + 1']}
                        tickFormatter={(value) => `${value.toFixed(1)}%`}
                        tickCount={8}
                        label={{ value: 'Max Drawdown', position: 'insideBottom', offset: -9 }}
                      />
                      <YAxis 
                        dataKey="return" 
                        name="Annual Return %" 
                        tickFormatter={(value) => `${value.toFixed(1)}%`}
                        label={{ value: 'Annual Return', angle: -90, position: 'insideLeft', offset: -8 }}
                      />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }}
                        labelFormatter={() => ''}
                        formatter={(value, name, props) => [
                          `${value}%`,
                          name === 'return' ? 'Annual Return' : 'Max Drawdown'
                        ]}
                        labelStyle={{ display: 'none' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white p-3 border rounded shadow-lg">
                                <p className="font-semibold mb-1">{data.fullName || data.name}</p>
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
                      <Scatter dataKey="return" fill="#10b981" />
                    </ScatterChart>
                  </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <PerformanceChart data={alphaData} cumulativeReturns={cumulativeReturns} />
          </TabsContent>

          <TabsContent value="leaderboard" className="space-y-4">
            <div className="grid" style={{gridTemplateColumns: "42.5% 1fr", gap: "60px"}}>
              {/* Left side - Leaderboard items */}
              <div className="flex-none">
                <div className="grid gap-3">
                  {alphaData.map((alpha, index) => (
                    <AlphaCard 
                      key={alpha.name} 
                      alpha={alpha} 
                      rank={index + 1}
                      onViewDetail={() => {
                        setSelectedAlpha(alpha.name)
                        setRightPanelMode('detail')
                      }}
                      onViewChart={() => {
                        setSelectedAlphaChart(alpha.name)
                        setRightPanelMode('chart')
                      }}
                    />
                  ))}
                </div>
              </div>
              
              {/* Right side - Detail/Chart panel */}
              <div className="flex-none">
                {rightPanelMode === 'detail' && selectedAlpha && nameToClassMap[selectedAlpha] && sourceCode[nameToClassMap[selectedAlpha]] ? (
                  <div className="bg-white rounded-lg border">
                    <div className="p-4 border-b bg-gray-50">
                      <div className="flex justify-between items-center">
                        <h3 className="text-lg font-semibold">Source Code</h3>
                        <button 
                          onClick={() => {
                            setSelectedAlpha(null)
                            setRightPanelMode(null)
                          }}
                          className="text-gray-500 hover:text-gray-700"
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                    <div className="p-4">
                      <CodeViewer 
                        alpha={sourceCode[nameToClassMap[selectedAlpha]]}
                        onClose={() => {
                          setSelectedAlpha(null)
                          setRightPanelMode(null)
                        }}
                      />
                    </div>
                  </div>
                ) : rightPanelMode === 'chart' && selectedAlphaChart ? (
                  <div className="bg-white rounded-lg border">
                    <div className="p-4 border-b bg-gray-50">
                      <div className="flex justify-between items-center">
                        <div>
                          <h3 className="text-lg font-semibold">{selectedAlphaChart?.replace(/_/g, ' ')}</h3>
                          <p className="text-sm text-gray-600 mt-1">Cumulative Returns Over Time</p>
                        </div>
                        <button 
                          onClick={() => {
                            setSelectedAlphaChart(null)
                            setRightPanelMode(null)
                          }}
                          className="text-gray-500 hover:text-gray-700"
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                    <div className="p-4">
                      {(() => {
                        const getStrategyData = () => {
                          if (!cumulativeReturns) return null
                          
                          const strategyKey = Object.keys(cumulativeReturns.strategies).find(key => 
                            cumulativeReturns.strategies[key].name === selectedAlphaChart
                          )
                          
                          if (!strategyKey) return null
                          
                          const strategy = cumulativeReturns.strategies[strategyKey]
                          return cumulativeReturns.dates.map((date, index) => ({
                            date,
                            return: strategy.cumulative_returns[index] || 0
                          }))
                        }

                        const chartData = getStrategyData()

                        return chartData ? (
                          <ResponsiveContainer width="100%" height={560}>
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
                        )
                      })()}
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-50 rounded-lg border-2 border-dashed border-gray-300 h-96 flex items-center justify-center">
                    <div className="text-center text-gray-500">
                      <div className="text-lg mb-2">No content selected</div>
                      <div className="text-sm">Click "View Detail" or "View Chart" on any strategy to see details here</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="source" className="space-y-4">
            {selectedAlpha && nameToClassMap[selectedAlpha] && sourceCode[nameToClassMap[selectedAlpha]] ? (
              <CodeViewer 
                alpha={sourceCode[nameToClassMap[selectedAlpha]]}
                onClose={() => setSelectedAlpha(null)}
              />
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle>Source Code Viewer</CardTitle>
                  <CardDescription>
                    Select a strategy from the leaderboard to view its source code
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-8 text-gray-500">
                    No strategy selected. Go to the Leaderboard tab and click "View Detail" on any strategy.
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="individual" className="space-y-4">
            {selectedAlphaChart ? (
              <AlphaChart 
                alpha={alphaData.find(a => a.name === selectedAlphaChart)!}
                cumulativeReturns={cumulativeReturns}
                onClose={() => setSelectedAlphaChart(null)}
              />
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle>Individual Alpha Chart</CardTitle>
                  <CardDescription>
                    Select a strategy from the leaderboard to view its individual performance chart
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-8 text-gray-500">
                    No strategy selected. Go to the Leaderboard tab and click "View Chart" on any strategy.
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}